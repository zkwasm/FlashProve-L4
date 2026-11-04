# Making AI computation easier to verify

When an AI service gives you an answer, you usually have to trust that it ran the model it promised. Anthropic's open-source FlashProve offers a way to check the computation: the service produces a mathematical proof alongside its output, which another computer can verify.

The [upstream version we used](../upstream/README.md) lists an NVIDIA H100 with 80 GB of GPU memory as its supported hardware for generating proofs. Its real Llama 4 Maverick example also calls for 330 GB of system memory and 245 GB of disk space. These are substantial requirements for individual developers and small teams.

**We deliberately chose a small model to make the complete workflow accessible on more modest hardware.** We adapted FlashProve to run TinyStories260K, a pretrained language model, on a single NVIDIA L4 with 24 GB of GPU memory. This required changes to hardware compatibility, memory use and the model's layout.

We recorded a fixed identity for the model, ran a prediction, generated its proof, and checked that proof independently on a Mac. The result also matched a separate reference calculation. Deliberately changing the input, output, proof or recorded model identity caused verification to fail.

The practical value is a lower barrier to experimenting with verifiable AI. We published the code, model, proof and instructions. Developers can check the result on an ordinary computer without a GPU, and use an L4 to generate a new proof themselves.

Anthropic provided the proof method. Our contribution is making a small-model demonstration work on L4 hardware and independently reproducible. This covers one step of text generation, not the large Maverick workload on an L4. The proof checks the calculation, not whether an answer is factually correct.

[View the project and verification instructions](../README.md).
