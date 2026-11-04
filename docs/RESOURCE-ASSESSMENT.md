# Resource measurements and planning for larger models

The project accumulated useful measurements and compatibility findings, but **no 0.7B model has
been tested**. This note separates results from the published TinyStories260K run, arithmetic
storage estimates, and the engineering work needed before sizing hardware for another model.

## Measurements from the completed run

The workload used one L4, one request, 256 input tokens, one next-token claim, and one one-time
leaf. The converted shape is width 256, head width 32, six layers, dense/shared FFN widths 512/256,
two zero routed experts, and vocabulary 512. These details matter when comparing another workload.

| Item | Recorded value | Interpretation |
|---|---:|---|
| GPU | NVIDIA L4, 23,034 MiB reported | Hardware that successfully ran this workload; not a measured minimum |
| Original checkpoint | 1,056,540 bytes | Original small pretrained model |
| Converted GGUF | 15,228,544 bytes | Expanded layout, including duplicated and zero weights |
| Converted tensor elements | 4,395,776 | Includes padding and zero experts; not independently learned parameters |
| Public key | 1,601,016 bytes | Generated key file |
| Public leaf | 388,720 bytes | One generated leaf file |
| Request proof | 5,600,372 bytes | One request's proof; key and leaf are separate |
| Mac independent verification | 0.963 seconds | Single run, not a repeated benchmark |
| Key engine memory-pool high-water mark | 603,979,776 bytes, or 576 MiB | One pool during commitment; not total GPU memory |
| Leaf engine memory-pool high-water mark | 268,435,456 bytes, or 256 MiB | One pool during leaf generation |
| Key slab allocation | 1,500,951,552 bytes | Separate allocation reported during key generation |
| Leaf slab allocation | 1,926,064,128 bytes | Separate allocation reported during leaf generation |
| Engine reservation in the L4 path | 8 GiB | Allocation policy, not measured usage or a hardware minimum |

The commitment log reports 2.7 seconds for its key-file phase and 0.2 seconds for its leaf-entry
phase. Those rounded phase timings are not total inference/proof-generation latency. Peaks from
different pools or phases must not be added as though all occurred simultaneously.

We did **not** capture a complete GPU peak for inference and proof generation, host peak RSS,
repeated latency distributions, or a hardware cost benchmark. A 24 GB L4 working successfully
does not establish that the workload needs 24 GB, or that a larger model will fit.

The values above can be regenerated from the included files and logs:

```sh
node scripts/summarize-resources.mts
```

See [resource-summary.json](../runs/resource-summary.json), the
[commitment log](../runs/real-l4-v8/commit.log),
[hardware record](../runs/real-l4-v8/hardware.txt), and
[verification results](../runs/real-l4-v8/verification/results.json).
Missing measurements are explicitly `null` in the JSON.

## Engineering findings that can save another developer time

| Constraint encountered | What it means for another port |
|---|---|
| Fixed 24 GiB engine reservation plus 2 GiB outside the slab | The original policy can reject an L4 even for tiny weights. We changed the reservation to 8 GiB for the L4 path; that setting still needs validation for a larger workload. |
| Oversized shared-memory kernels | Several kernels needed smaller buffers or existing/global-memory alternatives. Fitting the weights does not ensure that a kernel can launch. |
| cuBLAS rejected some integer matrix shapes | An exact integer fallback was necessary. Correctness passed 288 cases, including graph capture; its performance on larger shapes was not benchmarked. |
| Host CPU without SHA instruction extensions | The G2 host needed the existing portable SHA implementation. GPU compatibility alone was insufficient. |
| Weight-reader assumptions about model dimensions | We added the specific 256/32 width/head-width pair. This is not arbitrary-model support. |
| Dense model versus accepted routed-expert layout | The conversion uses shared FFNs and zero experts. Other architectures need a checked mapping of their operations and parameters. |
| Exact zero matrices | They exposed scale/norm edge cases. Zero branches must preserve both numeric behavior and proof constraints. |
| Minimum product-layer size | Our 64-token configuration failed a `mu >= 16` kernel condition; 256 tokens resolved it for this shape. This is not a universal 256-token minimum for every model. |
| Public attention table partitions and lookup grouping | Configurations accepted by initial planning could still fail GPU initialization. Full execution is necessary after configuration changes. |

The fixes and regression tests are in the
[upstream patch](../upstream-l4-small-model.patch),
[change inventory](SOURCE-CHANGES.json), and [backend logs](../runs/l4-backend/).

## What can be said about a hypothetical 0.7B model

For exactly 700 million scalar weights, the raw payload arithmetic is:

| Representation | Weight payload only, decimal GB |
|---|---:|
| FP32 | 2.80 GB |
| FP16 | 1.40 GB |
| INT8 | 0.70 GB |
| Packed INT4 | 0.35 GB |

These are calculations, **not measured model files or proof-memory requirements**. They exclude
quantization metadata, model conversion, padding, staging buffers, activations, commitment tables,
proof workspaces and driver/library allocations. They do not establish that a given representation
or architecture is supported by this port.

We cannot scale the tiny-model measurements linearly by parameter count. Layer widths, head
dimensions, vocabulary size, sequence length, batch size and proof configuration affect different
allocations. Padding and power-of-two boundaries can introduce jumps. Streaming can reduce
resident memory while adding passes and transfer time. Ordinary inference fitting in memory
does not establish that commitment and proof generation fit.

**Current answer to “will a 0.7B model run on one L4?”: not yet established.** There is also no
measured basis here for claiming it requires an H100 or a particular amount of host RAM.

## Work needed for a credible 0.7B hardware estimate

1. **Choose the exact checkpoint and workload.** Record architecture, layer/FFN/head dimensions,
   vocabulary, storage format, context length, batch size and which outputs will be proved.
2. **Check the computation mapping.** Confirm support for its normalization, activation, attention,
   positional encoding, biases and model layout. Extend readers and kernels where necessary;
   changing a parameter count in the existing config is insufficient.
3. **Measure conversion expansion and numerical behavior.** Generate tensor checksums, fit integer
   scales and test agreement with the original model on representative prompts. Record any padding
   or duplicated weights before calculating commitment dimensions.
4. **Plan and profile each stage separately.** Use the verifier's array/accounting planners, then
   measure actual commitment and inference/proof GPU peaks, host RSS, disk use, transfer volume,
   latency and proof size. Planning alone did not catch all failures in our small-model run.
5. **Complete the independent acceptance test.** Verify the new proof, compare with an independent
   integer reference, and reject changed inputs, outputs, proofs and commitments. Repeat profiling
   before publishing a recommended hardware configuration.

The present contribution gives another developer a tested L4 starting point and a concrete list
of compatibility risks. A validated 0.7B sizing guide would be a separate result requiring those
additional measurements.
