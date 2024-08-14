# AcoRec

Yılmazer, H., & Özel, S. A. (2023). Diverse but Relevant Recommendations with Continuous Ant Colony Optimization. Mathematics, 12(16), 2497. https://doi.org/10.3390/math12162497

# Abstract

This paper introduces a novel method called AcoRec, which employs an enhanced version of Continuous Ant Colony Optimization for hyper-parameter adjustment and integrates a non-deterministic model to generate diverse recommendation lists. AcoRec is designed for cold-start users and long-tail item recommendations by leveraging implicit data from collaborative filtering techniques. Continuous Ant Colony Optimization is revisited with the convenience and flexibility of deep learning solid methods and extended within the AcoRec model. The approach computes stochastic variations of item probability values based on the initial predictions derived from a selected item-similarity model. The structure of the AcoRec model enables efficient handling of high-dimensional data while maintaining an effective balance between diversity and high recall, leading to recommendation lists that are both varied and highly relevant to user tastes. Our results demonstrate that AcoRec outperforms existing state-of-the-art methods, including two random-walk models, a graph-based approach, a well-known vanilla autoencoder model, an ACO-based model, and baseline models with related similarity measures, across various evaluation scenarios. These evaluations employ well-known metrics to assess the quality of top-N recommendation lists, using popular datasets including MovieLens, Pinterest, and Netflix.

# Reproduce

This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to reproduce the results published by the author on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu) for code that leverages the GPU
- MATLAB/MOSEK/Stata licenses where applicable

# Instructions

## The computational environment (Docker image)

This capsule is private and its environment cannot be downloaded at this time. You will need to rebuild the environment locally.

> If there's any software requiring a license that needs to be run during the build stage, you'll need to make your license available. See [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information.

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag 7fa684b9-8b61-43b7-b3fc-5bb49c6f494c; cd ..
```

> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

## Running the capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm --gpus all \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  7fa684b9-8b61-43b7-b3fc-5bb49c6f494c bash run
```
