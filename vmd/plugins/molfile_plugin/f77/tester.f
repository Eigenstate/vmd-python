c     testing frontend for the molfile plugin fortran interface
c     $Id: tester.f,v 1.1 2006/03/10 22:48:49 johns Exp $
c     (c) 2006 Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>

      program molfile
      implicit none

      integer*4 natom, maxatom, handle(4), status
      parameter (maxatom=3000*3)
      real*4    xyz(maxatom), box(6)

      character infile*200, intype*10
      integer i,j

      print*,'molfile fortran tester v0.01'

C     set some default values
      infile = 'TRAJEC.dcd'
      intype = 'auto'
      natom  = -1
      handle(1) = -1
      handle(2) = -1
      handle(3) = -1
      handle(4) = -1
      
      print*,'filename: ', infile
      print*,'type:     ', intype

C     set up everything and 
C     register all static plugins
      call f77_molfile_init

      call f77_molfile_open_read(handle(1),natom,infile,intype)

      if (handle(1).lt.0) then
         print*,'file type unknown or not registered'
      else
         print*,'file successfully opened:'
         print*,'handle:',handle(1)
         print*,'natom: ',natom
      end if

      do i=1,2000
         status = 1   ! status=1 on entry means read
         call f77_molfile_read_next(handle(1),natom,xyz(1),box,status);
         print*,'read ',i,'  status:',status
         print*,'atom(1)', (xyz(j),j=1,3)
         print*,'atom(10)',(xyz(j),j=31,33)
         print*,'atom(100)',(xyz(j),j=301,303)
         print*,'box',box
         if(status.eq.0) go to 666
         status = 0   ! status=0 on entry means skip
         call f77_molfile_read_next(handle(1),natom,xyz,box,status);
         print*,'read ',i,'  status:',status
         if(status.eq.0) go to 666
      end do
 666  continue

      infile='li-nh3_4-end.pdb'
      intype='pdb'
      call f77_molfile_open_read(handle(2),natom,infile,intype)

      if (handle(2).lt.0) then
         print*,'file type unknown or not registered'
      else
         print*,'file successfully opened:'
         print*,'handle:',handle(2)
         print*,'natom: ',natom
      end if

      do i=1,2000
         status = 1   ! status=1 on entry means read
         call f77_molfile_read_next(handle(2),natom,xyz(1),box,status);
         print*,'read ',i,'  status:',status
         if(status.eq.0) go to 6666
         print*,'atom(1)',  (xyz(j),j=1,3)
         print*,'atom(10)', (xyz(j),j=31,33)
         print*,'atom(100)',(xyz(j),j=301,303)
         print*,'box',box
         status = 0   ! status=0 on entry means skip
         call f77_molfile_read_next(handle(2),natom,xyz,box,status);
         print*,'read ',i,'  status:',status
         if(status.eq.0) go to 6666
      end do
 6666 continue
      call f77_molfile_open_read(handle(3),natom,infile,intype)
      print*,'handle:',handle(3)

      call f77_molfile_close_read(handle(1),status)
      print*,'handle:',handle(1)
      call f77_molfile_open_read(handle(1),natom,infile,intype)
      print*,'handle:',handle(1)
      call f77_molfile_open_read(handle(4),natom,infile,intype)
      print*,'handle:',handle(4)


      call f77_molfile_close_read(handle(2),status)
      print*,'handle:',handle(2)
      call f77_molfile_close_read(handle(1),status)
      print*,'handle:',handle(1)
      call f77_molfile_close_read(handle(3),status)
      print*,'handle:',handle(3)
      call f77_molfile_close_read(handle(2),status)
      print*,'handle:',handle(2)
      call f77_molfile_close_read(handle(4),status)
      print*,'handle:',handle(4)

      call f77_molfile_finish

      end
