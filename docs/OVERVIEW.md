# Making AI computation easier to verify

When an AI service gives you an answer, you usually have to trust that it ran the model it promised. Anthropic's open-source FlashProve offers a way to check the computation: the service produces a mathematical proof alongside its output, which another computer can verify.

**We adapted FlashProve to run the complete process with a real, small pretrained model on a single NVIDIA L4 GPU.** The model was TinyStories260K, a small language model trained on simple stories. Our work addressed hardware compatibility and memory requirements, and adapted the model's structure so the existing proof system could process it.

We recorded a fixed identity for the model, ran a prediction, generated its proof, and checked that proof independently on a Mac. The result also matched a separate reference calculation. Deliberately changing the input, output, proof or recorded model identity caused verification to fail.

The practical value is a working example on more accessible hardware, with evidence that other people can inspect. We published the code, model, proof and instructions. Developers can check the published result on an ordinary computer without a GPU, and use an L4 to generate a new proof themselves.

Anthropic provided the underlying proof method. Our contribution is the hardware and model adaptation, the complete working demonstration, and its reproducibility. This demonstration covers one step of text generation by a very small model. It verifies the specified calculation; it does not establish that an AI answer is factually correct.

[View the project and verification instructions](../README.md).
