# Setup your Comet.ml environment

Most experiments we run utilize the comet_ml Python API, which easily creates and saves training plots using an intuitive online interface. We strongly recommend using Comet to train anything, but it should not be mandatory.

1. Ideally, you should create an account on [Comet.ml](https://www.comet.com/site/) in order to be able to view and save your different experiments.
2. Create an API key within your account so that you can create experiments using the Python SDK (done automatically within the project).
3. Finally, setup the API key following [Comet's documentation](https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#set-the-api-key). Several avenues are proposed, but we recommend simply putting the following line with your API key in your `.bashrc/.zshrc` or equivalent:

``` bash
export COMET_API_KEY="<your_comet_api_key>"
```

> Most of our training scripts have the `--use_comet` flag that you need to specify in order to log the metrics of the experiments you're running on their server.
