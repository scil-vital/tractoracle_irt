# Training your own oracle

> This page is currently being built to provide additional guidance. However, for general steps, you need the following:  
> 1. Track several millions of streamlines on different subjects with your favorite method (ideally more than one tracking algorithm).
> 2. Filter your streamlines with your reference method (we used RecobundlesX, extractor_flow or Verifyber in our release, but use whatever method you want). You want to have plausible and unplausible streamlines in all cases. The plausible streamlines will have a score of 1 and the unplausible streamlines will have a score of 0.
> 3. Compile your "annotated" streamlines into a training HDF5, validation HDF5 and testing HDF5. Using something like `tractoracle_irt/datasets/create_streamlines_dataset_separated.py`.
> 4. Train the oracle using `tractoracle_irt/trainers/tractoraclenet_train.py`.
>
> It is possible that at this time some bugs might be left unattended. If so, please create an issue and it should be fixed shortly.