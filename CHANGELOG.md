## [3.1.6] - 2025-01-28

### Changed

- Don't build dynamic plugins for molfile since they are all build statically
  as well. This removes a lot of spurious warnings.

### Fixed

- Build with Clang.

 
## [3.1.5] - 2025-01-27

### Fixed

- Support for Python <= 3.12.
- Initialize atomsel type with the spec-based initializer
instead of manually creating the type struct. This is part of
the Limited ABI so should be a lot more future proof. 

  
