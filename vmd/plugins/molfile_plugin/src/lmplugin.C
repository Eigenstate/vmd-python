/*
 * University of Illinois Open Source License
 * Copyright 2018-2018 Luthey-Schulten Group,
 * All rights reserved.
 *
 * Developed by: Luthey-Schulten Group
 * 			     University of Illinois at Urbana-Champaign
 * 			     http://www.scs.uiuc.edu/~schulten
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the Software), to deal with
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * - Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimers.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimers in the documentation
 * and/or other materials provided with the distribution.
 *
 * - Neither the names of the Luthey-Schulten Group, University of Illinois at
 * Urbana-Champaign, nor the names of its contributors may be used to endorse or
 * promote products derived from this Software without specific prior written
 * permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS WITH THE SOFTWARE.
 *
 * Author(s): Tyler M. Earnest
 */

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <exception>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "molfile_plugin.h"
#include "vmdconio.h"

namespace {

#define MAX_VMD_NAME 16
#define RDME_RADIUS_SCALE 0.1
#define TO_ANGSTROM 1e10
#define MAX_LMPEXC_MSG 256

typedef unsigned char particle_int_t;
typedef unsigned char site_int_t;

class LMPException : public std::exception {
protected:
    char msg[MAX_LMPEXC_MSG];
public:
    void vmdcon_report() { vmdcon_printf(VMDCON_ERROR, "LMPlugin) Error: %s\n", msg); }
};

class Exception : public LMPException {
public:
    Exception(const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
    }
};

class H5Exception : public LMPException {
    static herr_t
    walk_cb(unsigned int n,
            const H5E_error2_t *err,
            void *data)
    {
        if (n == 0) {
            snprintf(static_cast<char*>(data), MAX_LMPEXC_MSG, "HDF5: %s: %s",
                    err->func_name, err->desc);
        }
        return 0;
    }

public:
    H5Exception() { H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, H5Exception::walk_cb, msg); }
};

#define H5_CALL_ASSIGN(val, call)                                                                  \
    do {                                                                                           \
        val = (call);                                                                              \
        if (val < 0) {                                                                             \
            throw H5Exception();                                                                   \
        }                                                                                          \
    } while (0)

#define H5_CALL(call)                                                                              \
    do {                                                                                           \
        hid_t val;                                                                                 \
        H5_CALL_ASSIGN(val, (call));                                                               \
    } while (0)


struct Timer {
    // using low resolution time() to avoid portability issues
    time_t start, end;

    Timer() { reset(); }
    void reset() { time(&start); }

    int
    elapsed()
    {
        time(&end);
        return end-start;
    }
};

namespace H5File_specialized {

template <typename T> hid_t get_type_id();
template <> inline hid_t get_type_id<char>() { return H5T_STD_I8LE; }
template <> inline hid_t get_type_id<unsigned char>() { return H5T_STD_U8LE; }
template <> inline hid_t get_type_id<int>() { return H5T_STD_I32LE; }
template <> inline hid_t get_type_id<unsigned int>() { return H5T_STD_U32LE; }
template <> inline hid_t get_type_id<float>() { return H5T_IEEE_F32LE; }
template <> inline hid_t get_type_id<double>() { return H5T_IEEE_F64LE; }

template <typename T>
void
get_attr(hid_t loc_id,
         std::string obj_name,
         std::string attr_name,
         T &data)
{
    H5_CALL(H5LTget_attribute(loc_id, obj_name.c_str(), attr_name.c_str(), get_type_id<T>(),
                &data));
}

template <>
void
get_attr<std::string>(hid_t loc_id,
                      std::string obj_name,
                      std::string attr_name,
                      std::string &str)
{
    hid_t attr, type;
    hsize_t size;

    H5_CALL_ASSIGN(attr, H5Aopen_by_name(loc_id, obj_name.c_str(), attr_name.c_str(),
                H5P_DEFAULT, H5P_DEFAULT));
    H5_CALL_ASSIGN(type, H5Aget_type(attr));

    if (H5Tget_class(type) == H5T_STRING) {
        hid_t space;
        H5_CALL_ASSIGN(space, H5Aget_space(attr));
        H5_CALL_ASSIGN(size, H5Aget_storage_size(attr));

        hid_t memtype;
        H5_CALL_ASSIGN(memtype, H5Tcopy(H5T_C_S1));
        H5_CALL(H5Tset_size(memtype, size));

        char *data = new char[size + 1];
        H5_CALL(H5Aread(attr, memtype, data));
        data[size] = 0;
        str = std::string(data);
        delete [] data;

        H5_CALL(H5Sclose(space));
        H5_CALL(H5Tclose(memtype));
    }

    H5_CALL(H5Tclose(type));
    H5_CALL(H5Aclose(attr));
}

template <typename T>
bool
str_conv1(std::string str,
          T &val)
{
    std::istringstream ss(str);
    return ss >> val;
}

template <typename T>
bool
str_conv_last(std::string str,
              std::vector<T> &vals)
{
    return str.size() == 0;
}

// empty strings are valid
template <>
bool
str_conv1(std::string str,
          std::string &val)
{
    std::istringstream ss(str);
    ss >> val;
    return true;
}

template <>
bool
str_conv_last(std::string str,
              std::vector<std::string> &vals)
{
    if (str.size() > 0 && *(str.end()-1) == ',') {
        vals.push_back("");
    }
    return true;
}
} // end namespace H5File_specialized

class H5File {
    hid_t h5file;
    std::string filename;

    static herr_t
    get_group_list__proc(hid_t g_id,
                         const char *name,
                         const H5L_info_t *info,
                         void *op_data)
    {
        std::vector<std::string> *vec = static_cast<std::vector<std::string> *>(op_data);
        vec->push_back(name);
        return 0;
    }

public:

    H5File () : h5file(-1) {}
    ~H5File () { if (h5file>=0) { close(); } }

    void
    open(std::string filename_)
    {
        filename = filename_;
        H5_CALL_ASSIGN(h5file, H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    }

    void
    open_rdwr(std::string filename_)
    {
        filename = filename_;
        H5_CALL_ASSIGN(h5file, H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    }

    void
    close()
    {
        H5_CALL(H5Fclose(h5file));
        h5file = -1;
    }

    void
    get_userblock(std::vector<char>& data)
    {
        hid_t userblock;
        hsize_t userblockSize;
        H5_CALL_ASSIGN(userblock, H5Fget_create_plist(h5file));
        if (H5Pget_userblock(userblock, &userblockSize) < 0) {
            data.resize(0);
        }
        else {
            data.resize(userblockSize);
            close();
            FILE *fp;
            fp = fopen(filename.c_str(), "rb");
            fread(&data[0], userblockSize, 1, fp);
            fclose(fp);
            open(filename);
        }
    }

    template <typename T>
    void
    get_attr(std::string obj_name,
             std::string attr_name,
             T &data)
    {
        H5File_specialized::get_attr(h5file, obj_name, attr_name, data);
    }

    template <typename T>
    void
    get_csv_attr(std::string obj_name,
                 std::string attr_name,
                 std::vector<T> &data)
    {
        std::string str;
        H5File_specialized::get_attr(h5file, obj_name.c_str(), attr_name.c_str(), str);

        std::istringstream ss;
        ss.str(str);
        data.clear();

        std::string sval;

        while (std::getline(ss, sval, ',')) {
            T val;
            if (! H5File_specialized::str_conv1(sval, val) ) {
                throw Exception("String conversion failed for attribute %s on %s: Bad token: \"%s\"",
                        attr_name.c_str(), obj_name.c_str(), sval.c_str());
            }
            data.push_back(val);
        }

        if ( !H5File_specialized::str_conv_last(sval, data) ) {
            throw Exception("String conversion failed for attribute %s on %s: Bad token: \"%s\"",
                    attr_name.c_str(), obj_name.c_str(), sval.c_str());
        }
    }

    template <typename T>
    void
    get_dataset(std::string path,
                std::vector<T> &data,
                std::vector<hsize_t> &shape,
                hsize_t* nelemptr=NULL)
    {
        hid_t datasetHandle, dataspaceHandle, typeHandle, h5type;
        hsize_t dims[H5S_MAX_RANK];
        hsize_t ndims;

        H5_CALL_ASSIGN(datasetHandle, H5Dopen2(h5file, path.c_str(), H5P_DEFAULT));
        H5_CALL_ASSIGN(dataspaceHandle, H5Dget_space(datasetHandle));
        H5_CALL(H5Sget_simple_extent_dims(dataspaceHandle, dims, NULL));
        H5_CALL_ASSIGN(ndims, H5Sget_simple_extent_ndims(dataspaceHandle));
        H5_CALL_ASSIGN(typeHandle, H5Dget_type(datasetHandle));
        H5_CALL_ASSIGN(h5type, H5Tget_native_type(typeHandle, H5T_DIR_ASCEND));

        if (H5Tequal(H5File_specialized::get_type_id<T>(), h5type) <= 0) {
            throw Exception("Wrong HDF5 data type for ``%s''", path.c_str());
        }

        shape.resize(ndims);
        std::copy(dims, dims + ndims, shape.begin());

        hsize_t nelem = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<hsize_t>());

        if (nelemptr) {
            *nelemptr = nelem;
        }

        if (nelem != data.size()) {
            data.resize(nelem);
        }

        H5_CALL(H5Dread(datasetHandle, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(data[0])));

        H5_CALL(H5Tclose(typeHandle));
        H5_CALL(H5Sclose(dataspaceHandle));
        H5_CALL(H5Dclose(datasetHandle));
    }

    bool
    group_exists(std::string path)
    {
        H5E_auto2_t funcOld;
        void *dataOld;
        bool exists;

        H5Eget_auto2(H5E_DEFAULT, &funcOld, &dataOld);
        H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

        hid_t handle = H5Gopen2(h5file, path.c_str(), H5P_DEFAULT);

        exists = handle >= 0;
        if (exists) {
            H5Gclose(handle);
        }
        H5Eset_auto2(H5E_DEFAULT, funcOld, dataOld);
        return exists;
    }

    bool
    dataset_exists(std::string path)
    {
        H5E_auto2_t funcOld;
        void *dataOld;
        bool exists;

        H5Eget_auto2(H5E_DEFAULT, &funcOld, &dataOld);
        H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

        hid_t handle = H5Dopen2(h5file, path.c_str(), H5P_DEFAULT);

        exists = handle >= 0;
        if (exists) {
            H5Dclose(handle);
        }
        H5Eset_auto2(H5E_DEFAULT, funcOld, dataOld);
        return exists;
    }

    bool
    attr_exists(std::string path,
                std::string attr)
    {
        if (group_exists(path)) {
            hid_t objHandle = H5Oopen(h5file, path.c_str(), H5P_DEFAULT);
            int exists = H5LTfind_attribute(objHandle, attr.c_str());
            H5Oclose(objHandle);
            return exists;
        }
        else {
            return false;
        }
    }

    void
    get_group_listing(std::string path,
                      std::vector<std::string> &listing)
    {
        hid_t groupHandle;
        hsize_t ix = 0;
        listing.clear();
        H5_CALL_ASSIGN(groupHandle, H5Gopen(h5file, path.c_str(), H5P_DEFAULT));
        H5_CALL(H5Literate(groupHandle, H5_INDEX_NAME, H5_ITER_INC, &ix, get_group_list__proc, &listing));
        H5_CALL(H5Gclose(groupHandle));
    }

    template <typename T>
    void
    write_ds(std::string path,
             std::string name,
             const std::vector<hsize_t>& shape,
             const std::vector<T>& data)
    {
        hid_t groupHandle;
        hsize_t dims[H5S_MAX_RANK];
        std::copy(shape.begin(), shape.end(), dims);

        H5_CALL_ASSIGN(groupHandle, H5Gopen(h5file, path.c_str(), H5P_DEFAULT));
        H5_CALL(H5LTmake_dataset(groupHandle, name.c_str(), shape.size(), dims,
                    H5File_specialized::get_type_id<T>(), &(data[0])));
        H5_CALL(H5Gclose(groupHandle));
    }
};

class LMPlugin {
    H5File h5file;
    std::string filename;
    int replicate;

    std::vector<std::string> speciesNames, siteNames;
    std::vector<std::string> dsNames;

    unsigned int totalRdmeAtom, totalSiteAtom, nAtom;
    std::vector<unsigned int> nRdmeAtoms, nSiteAtoms;
    std::vector<unsigned int> rdmeAtomOffsets, siteAtomOffsets;

    std::vector<double> frameTimes;
    std::vector<particle_int_t> pLattice;
    std::vector<site_int_t> sLattice;

    std::vector<unsigned int> trajIndices;

    unsigned int nSpecies, nSiteType;
    unsigned int nLatticeX, nLatticeY, nLatticeZ, nLatticeP;
    float latticeSpacing;
    float rdmeRadius;
    unsigned int frameIx, nFrame;
    bool hasVariableSites, hasTrajData;

    int bondFrom, bondTo;

    static std::string
    disambiguate_name(std::string name,
                      int dupCt)
    {
        if (name.size() < MAX_VMD_NAME && dupCt == 0) {
            return name;
        }
        else {
            char shortName[MAX_VMD_NAME];
            int numlen = dupCt == 0 ? 1 : log10(dupCt) + 1;
            int j;

            for (j = 0; j < MAX_VMD_NAME - numlen - 2; j++) { // 2 = 1(separator)+1(null)
                shortName[j] = name[j];
            }
            snprintf(shortName + j, MAX_VMD_NAME, "~%d", dupCt);

            return std::string(shortName);
        }
    }

    void
    open_lm_file()
    {
        try {
            h5file.open(filename);
        }
        catch (H5Exception &exc) {
            throw Exception("Invalid LM file: Not an HDF5 file.");
        }
        std::vector<char> userblock;
        h5file.get_userblock(userblock);

        if (userblock.size() != 512 || strncmp("LMH5",  &userblock[0], 5) != 0) {
            throw Exception("Invalid LM file: Bad userblock");
        }
    }

    std::string
    sim_path(std::string relPath = "")
    {
        char path[128];
        snprintf(path, sizeof(path), "/Simulations/%07d/%s", replicate, relPath.c_str());
        return std::string(path);
    }

    void
    read_times()
    {
        std::vector<hsize_t> shape;

        h5file.get_dataset(sim_path("LatticeTimes"), frameTimes, shape);
        h5file.get_group_listing(sim_path("Lattice"), dsNames);

        // Verify site data consistent with rdme particle data
        if (hasVariableSites) {
            std::vector<double> times;
            std::vector<std::string> names;
            h5file.get_dataset(sim_path("SiteTimes"), times, shape);
            h5file.get_group_listing(sim_path("Sites"), names);
            if (dsNames != names || frameTimes != times) {
                throw Exception("Particle lattice and site lattice data appear to be inconsistent");
            }
        }

        nFrame = dsNames.size();
    }

    void
    read_parameters()
    {
        h5file.get_attr("/Model/Diffusion", "numberSiteTypes", nSiteType);
        h5file.get_attr("/Model/Diffusion", "numberSpecies", nSpecies);
        h5file.get_attr("/Model/Diffusion", "latticeSpacing", latticeSpacing);
        h5file.get_attr("/Model/Diffusion", "latticeXSize", nLatticeX);
        h5file.get_attr("/Model/Diffusion", "latticeYSize", nLatticeY);
        h5file.get_attr("/Model/Diffusion", "latticeZSize", nLatticeZ);
        h5file.get_attr("/Model/Diffusion", "particlesPerSite", nLatticeP);

        hasVariableSites = h5file.dataset_exists(sim_path("Sites/0000000001"));
        hasTrajData = h5file.group_exists(sim_path("Lattice"));

        latticeSpacing *= TO_ANGSTROM;
        rdmeRadius = RDME_RADIUS_SCALE * latticeSpacing;
    }

    template <typename T>
    void
    count_lattice_objects(std::string path,
                          std::vector<unsigned int>& count,
                          std::vector<T>& data,
                          std::vector<unsigned int>& tmp)
    {
        std::vector<hsize_t> shape;
        hsize_t nData;

        h5file.get_dataset(path, data, shape, &nData);

        std::fill(tmp.begin(), tmp.end(), 0);

        for (unsigned int pIx = 0; pIx < nData; pIx++) {
            T type = data[pIx];
            if (type != 0) {
                tmp[type]++;
            }
        }

        for (unsigned int i = 0; i < tmp.size(); i++) {
            count[i] = std::max(tmp[i], count[i]);
        }
    }

    void
    write_particle_counts()
    {
        h5file.close();
        h5file.open_rdwr(filename);

        std::vector<hsize_t> shape;
        shape.resize(1);

        shape[0] = nSpecies + 1;
        h5file.write_ds(sim_path(), "MaxParticleCounts", shape, nRdmeAtoms);

        shape[0] = nSiteType;
        h5file.write_ds(sim_path(), "MaxSiteCounts", shape, nSiteAtoms);

        h5file.close();
        h5file.open(filename);

        vmdcon_printf(
            VMDCON_INFO,
            "LMPlugin) Maximum particle counts written to %s:%s\n",
            filename.c_str(), sim_path().c_str());
    }

    void
    calculate_particle_counts()
    {
        std::vector<unsigned int> tmp;

        Timer timer;
        float t;

        // Count RDME max particles. If there is no trajectory stored in the file,
        // load the initial conditions. The maximum particle counts will not be
        // saved to the HDF5 file.
        tmp.resize(nSpecies + 1);

        timer.reset();
        if (hasTrajData) {
            vmdcon_printf(VMDCON_WARN,
                          "LMPlugin) Missing maximum particle count data. Rebuilding it now.\n");

            for (unsigned int fIx = 0, nproc = 0; fIx < dsNames.size(); fIx++, nproc++) {
                if ((t = timer.elapsed()) > 60) {
                    vmdcon_printf(VMDCON_INFO,
                                  "LMPlugin) %lu/%lu particle lattices processed (%.1f/sec).\n", fIx,
                                  dsNames.size(), nproc / t);
                    nproc = 0;
                    timer.reset();
                }

                count_lattice_objects(sim_path("Lattice/" + dsNames[fIx]), nRdmeAtoms, pLattice, tmp);
            }
        }
        else {
            count_lattice_objects("Model/Diffusion/Lattice", nRdmeAtoms, pLattice, tmp);
        }

        vmdcon_printf(VMDCON_INFO, "LMPlugin) RDME particles counted.\n");

        // Count site max particles. Use either the static site lattice or
        // variable data if available
        tmp.resize(nSiteType + 1);

        timer.reset();
        if (hasVariableSites) {
            for (unsigned int fIx = 0, nproc = 0; fIx < dsNames.size(); fIx++, nproc++) {
                if ((t = timer.elapsed()) > 60) {
                    vmdcon_printf(VMDCON_INFO,
                                  "LMPlugin) %lu/%lu site lattices processed (%.1f/sec).\n", fIx,
                                  dsNames.size(), nproc / t);
                    nproc = 0;
                    timer.reset();
                }
                count_lattice_objects(sim_path("Sites/" + dsNames[fIx]), nSiteAtoms, sLattice, tmp);
            }
            vmdcon_printf(VMDCON_INFO, "LMPlugin) Site particles counted (time dependent).\n");
        }
        else {
            count_lattice_objects("/Model/Diffusion/LatticeSites", nSiteAtoms, sLattice, tmp);
            vmdcon_printf(VMDCON_INFO, "LMPlugin) Site particles counted (constant).\n");
        }

        // Write histograms to the trajectory file so we don't have to do it again next time.
        if (hasTrajData) {
            write_particle_counts();
        }
    }

    void
    make_offsets(unsigned int &total,
                 std::vector<unsigned int> &maxCts,
                 std::vector<unsigned int> &offsets)
    {
        // The offset vector is constructed such that particle/site ids can be used as indexes.
        // Since we ignore the id=0 particles and sites, we explicitly set it to zero in case
        // they got counted anyway.
        maxCts[0] = 0;
        total = std::accumulate(maxCts.begin(), maxCts.end(), 0);

        /// partial_sum takes {a, b, c, d} to {a, a+b, a+b+c, a+b+c+d}. We need {0, a, a+b,
        /// a+b+c} to lookup the start of the atom data using particle id.
        offsets.resize(maxCts.size());
        offsets[0] = 0;
        std::partial_sum(maxCts.begin(), maxCts.end() - 1, offsets.begin() + 1);
    }

    void
    read_particle_counts()
    {
        std::vector<hsize_t> shape;

        nRdmeAtoms.resize(nSpecies + 1);
        std::fill(nRdmeAtoms.begin(), nRdmeAtoms.end(), 0);
        nSiteAtoms.resize(nSiteType + 1);
        std::fill(nSiteAtoms.begin(), nSiteAtoms.end(), 0);

        if (h5file.dataset_exists(sim_path("MaxParticleCounts")) ) {
            h5file.get_dataset(sim_path("MaxParticleCounts"), nRdmeAtoms, shape);
            h5file.get_dataset(sim_path("MaxSiteCounts"), nSiteAtoms, shape);
        }
        else {
            calculate_particle_counts();
        }

        // make offsets for each particle type into buffer provided by read_timestep
        make_offsets(totalRdmeAtom, nRdmeAtoms, rdmeAtomOffsets);
        make_offsets(totalSiteAtom, nSiteAtoms, siteAtomOffsets);

        // sites are placed after particles in the read_timestep buffer
        for (unsigned int i = 0; i < siteAtomOffsets.size(); i++) {
            siteAtomOffsets[i] += totalRdmeAtom;
        }

        nAtom = totalRdmeAtom + totalSiteAtom;

        // Extra element since the species counts do not include id=0
        trajIndices.resize(std::max(nSpecies + 1, nSiteType));
    }

    void
    load_and_fix_names(std::vector<std::string>& names,
                       const std::vector<std::string> &oldNames,
                       const char* type)
    {
        std::set<std::string> seen;

        for (unsigned int i = 0; i < oldNames.size(); i++) {
            std::string name = oldNames[i];
            std::string name0 = name;
            for (int dupCt = 0; seen.count(name) > 0; dupCt++) {
                name = disambiguate_name(name0, dupCt);
            }
            seen.insert(name);
            names.push_back(name);
            if (name != name0) {
                vmdcon_printf(VMDCON_WARN, "LMPlugin) %s ``%s'' renamed to ``%s''\n",
                              type, name0.c_str(), name.c_str());
            }
        }
    }

    void
    read_names()
    {
        std::vector<std::string> nameTmp;
        h5file.get_csv_attr("/Parameters", "speciesNames", nameTmp);
        speciesNames.push_back(""); // blank for id=0

        // ensure all names are unique after truncation to MAX_VMD_NAME characters
        load_and_fix_names(speciesNames, nameTmp, "Species");

        if (h5file.attr_exists("/Parameters", "siteTypeNames")) {
            h5file.get_csv_attr("/Parameters", "siteTypeNames", nameTmp);
        }
        else { // older LM files are missing this data
            nameTmp.clear();
            for (unsigned int i = 0; i < nSiteType; i++) {
                std::stringstream ss;
                ss << "site" << i;
                nameTmp.push_back(ss.str());
            }
        }

        load_and_fix_names(siteNames, nameTmp, "Site");
    }

    void
    define_molfile_atom(molfile_atom_t *&atomPtr,
                        unsigned int ix,
                        const char *segname,
                        const char *fmt,
                        float radius,
                        const std::vector<std::string >& names,
                        const std::vector<unsigned int>& counts)
    {
        unsigned int ncopies = counts[ix];
        if (ncopies > 0) {
            molfile_atom_t atomType;
            memset(&atomType, 0, sizeof(atomType));

            std::copy(names[ix].begin(), names[ix].end(), atomType.name);
            std::strcpy(atomType.segid, segname);
            snprintf(atomType.type, sizeof(atomType.type), fmt, ix);

            atomType.resid = 1;
            atomType.radius = radius;

            std::fill(atomPtr, atomPtr + ncopies, atomType);
            atomPtr += ncopies;
        }
    }

    static inline
    float
    lattice_jitter(unsigned int co,
                   int axis)
    {
        // We're using this custom prng so that particles which do not transition between subvolumes
        // between frames do not appear to move
        co = (3*co + axis)*0xabcd1234;

        // xorshift32
        co ^= co << 13;
        co ^= co >> 17;
        co ^= co << 5;
        co &= 0xffffffff;

        // Scale/translate the noise vector to keep the particles inside a subvolume.
        // Here, 2*RDME_RADIUS_SCALE*latticeSpacing = diameter.
        return 2.328306e-10f*(1.0f - 2.0f * RDME_RADIUS_SCALE) * co + RDME_RADIUS_SCALE;
    }

  public:
    LMPlugin(const char *file)
        : filename(file)
        , replicate(0)
        , totalRdmeAtom(0)
        , totalSiteAtom(0)
        , frameIx(0)
        , nFrame(0)
        , bondFrom(0)
        , bondTo(0)
    {
        if (getenv("LM_REPLICATE") != NULL) {
            replicate = atoi(getenv("LM_REPLICATE"));
        }

        // replicate 0 shouldn't exist, if atoi fails we'll get 0
        if (replicate <= 0) {
            replicate = 1;
        }

        open_lm_file();

        // set nSiteType, nSpecies, latticeSpacing, nLatticeX, nLatticeY, nLatticeZ, nLatticeP,
        // hasVariableSites, hasTrajData, rdmeRadius
        read_parameters();

        // set frameTimes, dsNames, nFrame
        if (hasTrajData) {
            read_times();
            vmdcon_printf(VMDCON_INFO, "LMPlugin) %s, replicate %d:\n", file, replicate);
            vmdcon_printf(VMDCON_INFO, "LMPlugin)    Lattice dimensions: %dx%dx%dx%d\n", nLatticeX, nLatticeY,
                          nLatticeZ, nLatticeP);
            vmdcon_printf(VMDCON_INFO, "LMPlugin)    %d frames, %.3f to %.3f seconds.\n", nFrame,
                          frameTimes[0], frameTimes[nFrame - 1]);
            if (hasVariableSites) {
                vmdcon_printf(VMDCON_INFO, "LMPlugin)    Contains site type trajectory\n");
            }
        }
        else {
            vmdcon_printf(VMDCON_INFO, "LMPlugin) %s, initial conditions (replicate %d missing)\n",
                          file, replicate);
            vmdcon_printf(VMDCON_INFO, "LMPlugin)    Lattice dimensions: %dx%dx%dx%d\n", nLatticeX, nLatticeY,
                          nLatticeZ, nLatticeP);
            frameTimes.push_back(0.0);
            nFrame = 1;
        }

        // set totalRdmeAtom, nRdmeAtoms, rdmeAtomOffsets,
        //     totalSiteAtom, nSiteAtoms, siteAtomOffsets,
        //     nAtom, trajIndices
        read_particle_counts();

        // set speciesNames, siteNames
        read_names();

        vmdcon_printf(VMDCON_INFO, "LMPlugin) Atom usage:\n");
        vmdcon_printf(VMDCON_INFO, "LMPlugin)    RDME species types: %3d (%d atoms used)\n",
                      nSpecies, totalRdmeAtom);
        vmdcon_printf(VMDCON_INFO, "LMPlugin)    Site types:         %3d (%d atoms used)\n",
                      nSiteType, totalSiteAtom);
    }

    unsigned int get_nframe() const { return nFrame; }
    unsigned int get_natoms() const { return nAtom; }

    int
    read_structure(int *optflags,
                   molfile_atom_t *atoms)
    {
        *optflags = MOLFILE_RADIUS;
        molfile_atom_t *atomPtr = atoms;

        // Segid field is used to distinguish particles (RDME) and sites (SITE).

        for (unsigned int sp = 0; sp < speciesNames.size(); sp++) {
            define_molfile_atom(atomPtr, sp, "RDME", "s%u", rdmeRadius, speciesNames, nRdmeAtoms);
        }

        for (unsigned int st = 0; st < siteNames.size(); st++) {
            define_molfile_atom(atomPtr, st, "SITE", "st%u", 0.5f*latticeSpacing, siteNames, nSiteAtoms);
        }
        return MOLFILE_SUCCESS;
    }

    int
    read_bonds(int *nbonds,
               int **from,
               int **to,
               float **bondorder,
               int **bondtype,
               int *nbondtypes,
               char ***bondtypename)
    {
        // Defined to prevent VMD from trying to create bonds on its own
        *nbonds = 0;
        *from = &(bondFrom);
        *to = &(bondTo);
        *bondorder = NULL;
        *bondtype = NULL;
        *nbondtypes = 0;
        *bondtypename = NULL;
        return MOLFILE_SUCCESS;
    }

    int
    read_timestep(molfile_timestep_t *ts)
    {
        if (frameIx >= nFrame) {
            return MOLFILE_EOF;
        }

        if (ts) { // ts can be null if we're skipping a frame
            ts->physical_time = frameTimes[frameIx];
            std::vector<hsize_t> shape;
            unsigned int ix;

            // Unused atoms banished to {-lambda, -lambda, -lambda}
            std::fill(ts->coords, ts->coords + 3 * nAtom, -1.0f * latticeSpacing);

            // RDME particles
            if (hasTrajData) {
                h5file.get_dataset(sim_path("Lattice/" + dsNames[frameIx]), pLattice, shape);
            }
            else {
                h5file.get_dataset("/Model/Diffusion/Lattice", pLattice, shape);
            }

            // Keep track of the offset of to the next empty slot for each particle type
            // using a histogram
            std::fill(trajIndices.begin(), trajIndices.end(), 0);

            ix = 0;
            for (unsigned int i = 0; i < shape[0]; i++) {
                for (unsigned int j = 0; j < shape[1]; j++) {
                    for (unsigned int k = 0; k < shape[2]; k++) {
                        for (unsigned int p = 0; p < shape[3]; p++, ix++) {
                            particle_int_t type = pLattice[ix];
                            if (type != 0) {
                                unsigned int offset = 3 * (rdmeAtomOffsets[type] + trajIndices[type]);
                                float *coord = ts->coords + offset;
                                // Semi-deterministic position randomization to break up moire
                                // pattern
                                coord[0] = latticeSpacing * (i + lattice_jitter(ix, 0));
                                coord[1] = latticeSpacing * (j + lattice_jitter(ix, 1));
                                coord[2] = latticeSpacing * (k + lattice_jitter(ix, 2));
                                trajIndices[type]++;
                            }
                        }
                    }
                }
            }

            // site types
            if (hasVariableSites) {
                h5file.get_dataset(sim_path("Sites/" + dsNames[frameIx]), sLattice, shape);
            }
            else {
                h5file.get_dataset("/Model/Diffusion/LatticeSites", sLattice, shape);
            }

            std::fill(trajIndices.begin(), trajIndices.end(), 0);

            ix = 0;
            for (unsigned int i = 0; i < shape[0]; i++) {
                for (unsigned int j = 0; j < shape[1]; j++) {
                    for (unsigned int k = 0; k < shape[2]; k++, ix++) {
                        site_int_t type = sLattice[ix];
                        if (type != 0) {
                            unsigned int offset = 3 * (siteAtomOffsets[type] + trajIndices[type]);
                            float *coord = ts->coords + offset;
                            coord[0] = latticeSpacing * (i + 0.5f);
                            coord[1] = latticeSpacing * (j + 0.5f);
                            coord[2] = latticeSpacing * (k + 0.5f);
                            trajIndices[type]++;
                        }
                    }
                }
            }
        }

        frameIx++;
        return MOLFILE_SUCCESS;
    }
};
} // end anonymous namespace

static void *
open_read(const char *filename,
          const char *filetype,
          int *nAtom)
{
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    LMPlugin *lmd = NULL;
    try {
        lmd = new LMPlugin(filename);
        *nAtom = lmd->get_natoms();
        return lmd;
    }
    catch (LMPException &e) {
        e.vmdcon_report();
        return NULL;
    }
}

static int
read_structure(void *mydata,
               int *optflags,
               molfile_atom_t *atoms)
{
    LMPlugin *lmd = static_cast<LMPlugin *>(mydata);
    try {
        return lmd->read_structure(optflags, atoms);
    }
    catch (LMPException &e) {
        e.vmdcon_report();
        return MOLFILE_ERROR;
    }
}

static int
read_bonds(void *mydata,
           int *nbonds,
           int **from,
           int **to,
           float **bondorder,
           int **bondtype,
           int *nbondtypes,
           char ***bondtypename)
{
    LMPlugin *lmd = static_cast<LMPlugin *>(mydata);
    try {
        return lmd->read_bonds(nbonds, from, to, bondorder, bondtype, nbondtypes, bondtypename);
    }
    catch (LMPException &e) {
        e.vmdcon_report();
        return MOLFILE_ERROR;
    }
}

static int
read_timestep(void *mydata,
              int nAtom,
              molfile_timestep_t *ts)
{
    LMPlugin *lmd = static_cast<LMPlugin *>(mydata);
    try {
        return lmd->read_timestep(ts);
    }
    catch (LMPException &e) {
        e.vmdcon_report();
        return MOLFILE_ERROR;
    }
}

static void
close_read(void *mydata)
{
    LMPlugin *lmd = static_cast<LMPlugin *>(mydata);
    delete lmd;
}

VMDPLUGIN_API int
VMDPLUGIN_init()
{
    memset(&plugin, 0, sizeof(molfile_plugin_t));
    plugin.abiversion = vmdplugin_ABIVERSION;
    plugin.type = MOLFILE_PLUGIN_TYPE;
    plugin.name = "LM";
    plugin.prettyname = "Lattice Microbes (RDME)";
    plugin.author = "Tyler M. Earnest";
    plugin.majorv = 1;
    plugin.minorv = 0;
    plugin.is_reentrant = VMDPLUGIN_THREADSAFE;
    plugin.filename_extension = "lm";
    plugin.open_file_read = open_read;
    plugin.read_structure = read_structure;
    plugin.read_bonds = read_bonds;
    plugin.read_next_timestep = read_timestep;
    plugin.close_file_read = close_read;
    return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int
VMDPLUGIN_register(void *v,
                   vmdplugin_register_cb cb)
{
    (*cb)(v, (vmdplugin_t *) &plugin);
    return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int
VMDPLUGIN_fini()
{
    return VMDPLUGIN_SUCCESS;
}

#ifdef LMP_MAIN
int
main(int argc,
     char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s test.lm\n", argv[0]);
        return 1;
    }
    VMDPLUGIN_init();
    int nAtom, flags;
    LMPlugin *lmp = static_cast<LMPlugin *>(open_read(argv[1], NULL, &nAtom));
    if (!lmp) {
        fprintf(stderr, "%s: open_read() failed\n", argv[1]);
        return 1;
    }
    molfile_atom_t *atoms = new molfile_atom_t[nAtom];
    if (read_structure(lmp, &flags, atoms) != MOLFILE_SUCCESS) {
        fprintf(stderr, "%s: read_structure() failed\n", argv[1]);
        return 1;
    }

    int nbonds;
    int *from, *to, *bondtype, nbondtypes;
    float *bondorder;
    char **bondtypename;
    if (read_bonds(lmp, &nbonds, &from, &to, &bondorder, &bondtype, &nbondtypes, &bondtypename)
            != MOLFILE_SUCCESS) {
        fprintf(stderr, "%s: read_bonds() failed\n", argv[1]);
        return 1;
    }

    molfile_timestep_t trajdata;
    trajdata.coords = new float[3*nAtom];

    int stride = lmp->get_nframe()/3;
    stride = stride > 0 ? stride : 1;
    int nframes = stride*((lmp->get_nframe() + stride - 1)/stride);

    for (unsigned int i=0; i < nframes; i+=stride) {
        if (read_timestep(lmp, nAtom, &trajdata) != MOLFILE_SUCCESS) {
            fprintf(stderr, "%s: read_timestep(), frame index %d failed\n", argv[1], i);
            return 1;
        }
        printf("frame %d/%d\n", i+1, lmp->get_nframe());
    }

    close_read(lmp);
    delete [] trajdata.coords;
    delete [] atoms;
    return 0;
}
#endif
