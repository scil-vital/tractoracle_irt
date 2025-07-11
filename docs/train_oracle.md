# Training your own oracle

## 0. Prepare your data

To train your oracle, you'll need to have your data compiled into an HDF5 file as described by [this guide](./build_dataset_streamlines.md). If you haven't created your dataset, please follow [the guide](./build_dataset_streamlines.md). Once that is done, you can continue with the next steps below.

## 1. Train the oracle

At this point, you should have a single HDF5 file, that we will call `dataset_example.hdf5` for the sake of this guide. You can now train your oracle using the following script:

``` bash
python tractoracle_irt/trainers/tractoraclenet_train.py \
        <experiment_path> \
        <experiment_name> \
        <experiment_id> \
        <nb_epochs> \
        <path/to/dataset>/dataset_example.hdf5 \
        [--use_comet]
```
To visualize the training plots make sure you have [**properly set up Comet**](./setup_comet.md). This script will train for the specified number of epochs, will log everything into Comet.ml, and you'll be able to use the checkpoint of that oracle that is saved under the experiment path you provided when calling the training script.

For more information on more arguments you can supply to this script, simply run:
``` bash
python tractoracle_irt/trainers/tractoraclenet_train.py --help
```

# Next step
If you want to train your RL agent, you can leverage the checkpoint of the oracle you just trained to supply this into the RL training pipeline. [See this section to train your agent](../README.md#training-an-agent).