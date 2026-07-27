pytorch

For those on the call, here is the Helion on TPU blog post I mentioned today: [https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/](https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/)

For those who couldn't make it, you can still try out TorchTPU\! ![😄][image1]. The easiest way is to spin up a Colab with a v6e (if you can get it, although v5e also works) and pip install the libraries from this getting started guide: [TorchTPU GDE Quickstart](https://docs.google.com/document/d/1cvjk5_W2SlNJX0sAL0kXVxo1LrbZD4zACysGAKiPhIc)

Below this email is a list of resources to get you started and to dig deeper.

I'm excited to see everything you try\! And I'm here to help if something doesn't work. 

If you'd like to meet 1:1, that's great too \- here are good times I'm available: [https://calendar.google.com/appointments/schedules/AcZssZ2EKN-H2wmJfDbgeCqR8L4CnXbkl61jFH-mhSRVrU1FHRH6ySldElzV2dD50X9zwBgZTn\_8WRwu](https://calendar.google.com/appointments/schedules/AcZssZ2EKN-H2wmJfDbgeCqR8L4CnXbkl61jFH-mhSRVrU1FHRH6ySldElzV2dD50X9zwBgZTn_8WRwu)

thanks\!  
Chris

**TorchTPU**

* TorchTPU GDE Quickstart: [TorchTPU GDE Quickstart](https://docs.google.com/document/d/1cvjk5_W2SlNJX0sAL0kXVxo1LrbZD4zACysGAKiPhIc)  
* Repo: [https://github.com/google-pytorch/torch\_tpu](https://github.com/google-pytorch/torch_tpu)  
* Docs: [http://google-pytorch.github.io/torch\_tpu/](http://google-pytorch.github.io/torch_tpu/)  
* Overview video: [https://www.youtube.com/watch?v=H8SjVNB7YhM](https://www.youtube.com/watch?v=H8SjVNB7YhM)

**TPUs**

* **TPU Developers Hub** [https://cloud.google.com/products/tpu/tpu-developer](https://cloud.google.com/products/tpu/tpu-developer)  
* Section 2 of “How to Scale Your Model”, “How to think about TPUs”: [https://jax-ml.github.io/scaling-book/tpus/](https://jax-ml.github.io/scaling-book/tpus/)

**Tips for using PyTorch on TPUs**

* torch.compile(model, backend="tpu") can help maximize flops utilization  
* Avoid Graph Breaks: Avoid mixing Python control flow (if/else on tensor values), scalar-to-tensor conversions (.item()), or print statements inside compiled model forward passes.  
* Set Batch Sizes to Multiples of 8/128: Align tensor dimensions to multiples of 128 (or 8/16) to fit TPU Matrix Multiply Unit (MXU) systolic array tiles cleanly. The How to Scale Your Model book covers that.  
* Use BFloat16 Precision: Native torch.bfloat16 precision on MXU. float32 can be emulated  
* If you use an agent, point them to the tpu developer’s hub, and our documentation. Tell them not to use PyTorch/XLA or “import torch\_tpu” if you see it.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAARCAYAAAA7bUf6AAAB40lEQVR4XrVTSy9DURBuIkEEIcLCc1G0TR+6srRgLZFI2PoBEt01rCy7EfFKdEOiFSGhURGpDeodIgSREI+9PrX39tYjY2autre3WDHJl5Mz8803c+aco9H8l8Xd2lFhuRnEDQOIPgOvwkozxOcbbWpujsXntDbRqwfJbwTp0ATJY4KZV9pLu0YQ1/SAvC51LhsGZqmqdICJp2Z4vbDA2yXiSl5pT36Ki5sGEupTC5SJqzomvJ6Z4e3aAu+3LfDxYE3j/a6F/RRnIa8OskVc2scEHoEqscA9Jj7lgvwUJx7xgxMFnrSIsNQkd3FuAUd/NbQbSxhKgZTP1lPFPOJH56oy3dCwkkcmPvvkYC1EsdrkUG2WiGOgWvZjnHjEjy03KETWMyI8g2+OkgbGWQRvLO5RiuCQSCTiLoWwqxjP3JibjBD99RB05oPgq2N+1F2ZEREWcSb7JvAP5cOUww6/GcWjC+U8k8BYXkYEr9iT2DFCCJEa4G9InuDtIPd5VJP9gumZUzc3Kzom2jtMELTb0ui0VrA/hK+WeIFxRRcpQ1Wr4JWFuJKvDV6cvRAZ6YPYTDdIe63sp3hwupC6KFNrsFEg5CyCxLZcjf/P0de/wX3MU0PJYXXej4bkYcSWAsNqzp/aJ7e0A+dFT1n6AAAAAElFTkSuQmCC>