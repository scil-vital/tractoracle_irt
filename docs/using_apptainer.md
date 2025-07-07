# Using Apptainer instead of Docker for IRT

> TODO: 
> This guide is currently being built. If you require assistance with this guide, please open an issue.  
> In essence, some systems do not allow the use of docker containers (for security issues). In that scenario, using apptainer is generally a good alternative. All you have to do is build those docker images into singularity images and provide the path to the `.sif` files in appropriate places (e.g. RbxFilterer, ExtractorFilterer, VerifyberFilterer). This way, you'll be able to run the IRT scheme without depending on Docker containers.