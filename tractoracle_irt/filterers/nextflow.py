import os

def build_pipeline_command(pipeline: str, use_docker: bool, img_path: str, path_only: bool = False) -> str:
    """
    Build the nextflow command to run a pipeline.
    As most nextflow pipelines we use are designed to run in a containerized environment,
    this function will build the command accordingly.

    This is a sort of workaround of the nextflowpy library which does not support
    specifying the -with-docker or -with-singularity options directly in the command.

    Args:
        pipeline (str): Path of the pipeline to run. It either points to a GitHub
                        repository holding a main.nf file or directly towards
                        the main.nf file available locally.
        use_docker (bool): Use docker container if True, otherwise use Singularity/Apptainer.
        img_path (str): If using Singularity/Apptainer, the path to the image file to use.
                        If using Docker, this is the name of the Docker image to use.

    Returns:
        str: The command to run the RBX pipeline.
    """
    if path_only:
        return pipeline
    
    if use_docker:
        return f"{pipeline} -with-docker {img_path}"
    else:
        if not img_path:
            raise ValueError("When using Singularity/Apptainer, the img_path must be provided.")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"The specified Singularity/Apptainer image does not exist: {img_path}")
        return f"{pipeline} -with-singularity {img_path}"

