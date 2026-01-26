import os
import tempfile
from keopscore.config import config
from keopscore.utils.code_gen_utils import get_hash_name
from keopscore.utils.misc_utils import KeOps_Error, KeOps_Message, KeOps_Warning

cpp_flags = config.get_cpp_flags()
get_build_folder = config.get_build_folder


class LinkCompile:
    """
    Base class for compiling the map_reduce schemes and providing the dll to KeOps bindings.

    CACHE ATOMICITY FIX:
    - The cache is only considered valid when BOTH the compiled file AND the info file exist
    - Uses atomic file writes (write to temp, then rename) for the info file
    - Detects and cleans up partial/corrupt cache entries
    """

    def __init__(self, lang=None):
        # --- JAX MULTI-GPU PATCH START ---
        # If the JAX-specific flag is set, we force the CMake backend.
        self.jax_mode = (lang == "jax")

        if self.jax_mode:
            # Note: We no longer set config.use_nvrtc globally - backend selection
            # is now done per-class in the GpuReduc* files based on lang parameter
            KeOps_Message("JAX mode: using CMake backend for multi-GPU support")

        # --- JAX MULTI-GPU PATCH END ---

        # N.B. Here self is assumed to be populated by the __init__ of one of the MapReduce classes

        # # Get current cpp flags for hash calculation
        current_cpp_flags = config.get_cpp_flags()

        # we create the hash string id corresponding to all parameters
        self.gencode_filename = get_hash_name(
            type(self),
            self.red_formula,
            self.red_formula_string,
            self.aliases,
            self.nargs,
            self.dtype,
            self.dtypeacc,
            self.sum_scheme_string,
            self.tagHostDevice,
            self.tagCpuGpu,
            self.tag1D2D,
            self.use_half,
            self.use_fast_math,
            self.device_id,
            current_cpp_flags,  # Use updated flags
            lang,  # Include lang in hash for cache separation
        )

        # --- JAX MULTI-GPU PATCH START ---
        # Append suffix to ensure we don't accidentally load a PyTorch NVRTC cache file
        if self.jax_mode:
            self.gencode_filename += "_jax"
        # --- JAX MULTI-GPU PATCH END ---

        # info_file is the name of the file that will contain some meta-information
        self.info_file = os.path.join(
            get_build_folder(), self.gencode_filename + ".nfo"
        )

        # gencode_file is the name of the source file to be created
        self.gencode_file = os.path.join(
            get_build_folder(),
            self.gencode_filename + "." + self.source_code_extension,
        )

    def _is_cache_valid(self):
        """
        Check if the cache entry is complete and valid.

        A cache entry is only valid if BOTH:
        1. The compiled file (file_to_check) exists
        2. The info file (info_file) exists

        This prevents the case where the process was killed after compilation
        but before the info file was written.
        """
        file_exists = os.path.exists(self.file_to_check)
        info_exists = os.path.exists(self.info_file)

        if file_exists and not info_exists:
            # Partial cache entry detected - compiled file exists but info file is missing
            # This can happen if the process was killed during compilation
            KeOps_Warning(
                f"Detected incomplete cache entry: {self.gencode_filename}\n"
                f"  Compiled file exists: {self.file_to_check}\n"
                f"  Info file missing: {self.info_file}\n"
                f"  Cleaning up and recompiling..."
            )
            self._cleanup_partial_cache()
            return False

        if not file_exists and info_exists:
            # Info file exists but compiled file is missing (should be rare)
            KeOps_Warning(
                f"Detected incomplete cache entry: {self.gencode_filename}\n"
                f"  Compiled file missing: {self.file_to_check}\n"
                f"  Info file exists: {self.info_file}\n"
                f"  Cleaning up and recompiling..."
            )
            self._cleanup_partial_cache()
            return False

        return file_exists and info_exists

    def _cleanup_partial_cache(self):
        """
        Clean up partial/corrupt cache entries.

        Removes all files associated with this cache entry to ensure
        a clean recompilation.
        """
        files_to_remove = [
            self.file_to_check,
            self.info_file,
            self.gencode_file,
        ]

        # Also try to remove any other potential associated files
        # (e.g., .so.info files used by some backends)
        if hasattr(self, 'so_file'):
            files_to_remove.append(self.so_file)
            files_to_remove.append(self.so_file + ".info")

        if hasattr(self, 'low_level_code_file') and isinstance(self.low_level_code_file, str):
            files_to_remove.append(self.low_level_code_file)

        for filepath in files_to_remove:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as e:
                    KeOps_Warning(f"  Failed to remove {filepath}: {e}")

    def save_info(self):
        """
        Create info_file to save some parameters.

        Uses atomic write: writes to a temp file first, then renames.
        This ensures the info file either fully exists or doesn't exist at all,
        preventing partial/corrupt info files.
        """
        info_content = f"red_formula={self.red_formula_string}\ndim={self.dim}\ntagI={self.tagI}\ndimy={self.dimy}"

        # Get the directory for the info file
        info_dir = os.path.dirname(self.info_file)

        # Write to a temp file first, then atomically rename
        # This ensures the info file is never in a partial state
        fd, temp_path = tempfile.mkstemp(dir=info_dir, prefix=".tmp_info_", suffix=".nfo")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(info_content)
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, self.info_file)
        except Exception as e:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise e

    def read_info(self):
        """
        Read info_file to retrieve dim, tagI, dimy.

        Includes error handling for corrupt/incomplete info files.
        """
        try:
            with open(self.info_file, "r") as f:
                string = f.read()
        except FileNotFoundError:
            KeOps_Error(
                f"Info file not found: {self.info_file}\n"
                f"This may indicate an incomplete cache entry. "
                f"Try clearing the cache: rm -rf ~/.cache/keops*"
            )
        except IOError as e:
            KeOps_Error(f"Failed to read info file {self.info_file}: {e}")

        tmp = string.split("\n")
        if len(tmp) != 4:
            # Corrupt info file - clean up and report
            self._cleanup_partial_cache()
            KeOps_Error(
                f"Incorrect info file format in {self.info_file} "
                f"(expected 4 lines, got {len(tmp)}). "
                f"Cache entry has been cleaned up. Please retry your operation."
            )

        tmp_dim, tmp_tag, tmp_dimy = (
            tmp[1].split("="),
            tmp[2].split("="),
            tmp[3].split("="),
        )
        if (
                len(tmp_dim) != 2
                or tmp_dim[0] != "dim"
                or len(tmp_tag) != 2
                or tmp_tag[0] != "tagI"
                or len(tmp_dimy) != 2
                or tmp_dimy[0] != "dimy"
        ):
            # Corrupt info file - clean up and report
            self._cleanup_partial_cache()
            KeOps_Error(
                f"Incorrect info file content in {self.info_file}. "
                f"Cache entry has been cleaned up. Please retry your operation."
            )

        self.dim = eval(tmp_dim[1])
        self.tagI = eval(tmp_tag[1])
        self.dimy = eval(tmp_dimy[1])

    def write_code(self):
        # write the generated code in the source file
        with open(self.gencode_file, "w") as f:
            f.write(self.code)

    def generate_code(self):
        pass

    def get_dll_and_params(self):
        """
        Main method of the class: generates the code if needed.

        FIX: Uses _is_cache_valid() which checks for BOTH the compiled file
        AND the info file, preventing errors from incomplete cache entries.
        """
        if not self._is_cache_valid():
            KeOps_Message(
                "Generating code for " + self.red_formula.__str__() + " ... ",
                flush=True,
                end="",
            )
            self.generate_code()
            self.save_info()
            KeOps_Message("OK", use_tag=False, flush=True)
        else:
            self.read_info()
        return dict(
            tag=self.gencode_filename,
            source_file=self.true_dllname,
            low_level_code_file=self.low_level_code_file,
            tagI=self.tagI,
            use_half=self.use_half,
            use_fast_math=self.use_fast_math,
            tag1D2D=self.tag1D2D,
            dimred=self.red_formula.dimred,
            dim=self.dim,
            dimy=self.dimy,
            indsi=self.varloader.indsi,
            indsj=self.varloader.indsj,
            indsp=self.varloader.indsp,
            dimsx=self.varloader.dimsx,
            dimsy=self.varloader.dimsy,
            dimsp=self.varloader.dimsp,
        )
