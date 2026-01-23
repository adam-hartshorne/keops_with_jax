import os
from keopscore.config import config
from keopscore.utils.code_gen_utils import get_hash_name
from keopscore.utils.misc_utils import KeOps_Error, KeOps_Message

cpp_flags = config.get_cpp_flags()
get_build_folder = config.get_build_folder


class LinkCompile:
    """
    Base class for compiling the map_reduce schemes and providing the dll to KeOps bindings.
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
            current_cpp_flags, # Use updated flags
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

    def save_info(self):
        # create info_file to save some parameters
        with open(self.info_file, "w") as f:
            f.write(
                f"red_formula={self.red_formula_string}\ndim={self.dim}\ntagI={self.tagI}\ndimy={self.dimy}"
            )

    def read_info(self):
        # read info_file to retreive dim, tagI, dimy
        with open(self.info_file, "r") as f:
            string = f.read()

        tmp = string.split("\n")
        if len(tmp) != 4:
            KeOps_Error("Incorrect info file")
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
            KeOps_Error("Incorrect info file")
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
        # main method of the class : it generates - if needed - the code
        if not os.path.exists(self.file_to_check):
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
