<!--
{
  "title": "Erasing the Invisible Challenge",
  "description": "My first competition experience",
  "keywords": "AI, Machine Learning, Trustworthy AI, Image Watermarking, NeurIPS 2024, TreeRing, StegaStamp",
  "image": "erasing-the-invisible.jpg",
  "type": "article",
  "author": "Mohammad Jafari",
  "date": "2024-11-20",
  "category": "Machine Learning",
  "tags": ["AI", "Deep Learning", "Trustworthy AI"],
  "readingTime": "5 min"
}
-->

# My First Competition Experience

![Cover Image](erasing-the-invisible.jpg)

This was my first time attending a challenge, and it was such a **fun and rewarding experience**! I am so happy that I could end up as one of the winners of the **Erasing the Invisible NeurIPS2024 Challenge**. Here, I’d love to share some of the details of my approach, the challenges I faced, and how I solved them.

---

## What was the Competition About?

The **"Erasing the Invisible"** challenge at NeurIPS 2024 was an exciting competition focused on testing the resilience of image watermarks. The goal was to develop methods to **remove invisible watermarks embedded in images** while maintaining as much of the **original image quality** as possible. This was not just about breaking the watermark but also about ensuring that the resulting images looked **natural and unaltered**. The competition evaluated participants on two primary criteria:

- **How effectively the watermark was removed**.
- **How well the image quality was preserved**.

![Overview](overview.png)

---

## Setting Up the Pipeline

First things first, we need to set up the core components of the attack pipeline. This part might seem a bit routine, but it's absolutely crucial for what comes next.

I used a few key models for this attack, and each played a vital role in how the pipeline worked. Let’s dive into them:

### The Base Model

The backbone of the entire pipeline is the base model, `black-forest-labs/FLUX.1-dev`. This model acts as the foundation, providing the key building blocks for all the subsequent transformations. The choice of the base model impacts every stage of the attack, and I found **`FLUX.1-dev`** to strike a great balance between flexibility and quality.

### ControlNet Model for Edge Integrity

To help maintain edge integrity during manipulation, I used a **Canny edge detector**, specifically `InstantX/FLUX.1-dev-Controlnet-Canny`. Maintaining the core structure of an image is essential when making subtle changes, and this is where **ControlNet** really shines. It’s like a safety net ensuring that even with a strong diffusion process, the core composition of an image stays intact.

### Transformer: Keeping It Light

Due to GPU limitations, I opted for the **4-bit version** of the model (`sayakpaul/flux.1-dev-nf4-pkg`). My hardware couldn’t handle the full version, so using this quantized model was the way to go. It let me work effectively without the hardware becoming a roadblock.

### Loading the Models

Here’s how I loaded these models into our pipeline:

```python
controlnet_canny = FluxControlNetModel.from_pretrained(
    'InstantX/FLUX.1-dev-Controlnet-Canny',
    cache_dir='./models/',
    torch_dtype=torch.float16
)

ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"
transformer_4bit = FluxTransformer2DModel.from_pretrained(
    ckpt_4bit_id,
    subfolder="transformer",
    cache_dir='./models/'
)

base_model = 'black-forest-labs/FLUX.1-dev'
pipeline = FluxControlNetImg2ImgPipeline.from_pretrained(
    base_model,
    transformer=transformer_4bit,
    text_encoder=None,
    tokenizer=None,
    text_encoder_2=None,
    tokenizer_2=None,
    controlnet=controlnet_canny,
    torch_dtype=torch.float16,
    cache_dir='./models/'
)
```

#### Why Use ControlNet?

**ControlNet** plays a key role in ensuring that the composition of an image remains intact, even while undergoing significant manipulation. Without ControlNet, high-strength transformations can easily disrupt the composition, leading to undesirable outcomes. ControlNet essentially provides a way to guide the transformations more intelligently, making sure the final output doesn’t lose critical features of the original image.

#### My Model Choice Journey

Getting to the final model setup wasn’t straightforward. I started with image-based diffusion models like **GLIDE** and later moved on to **v-diffusions**. After two weeks of experimenting with different schedulers and samplers, I made the jump to **latent diffusion models**, despite their complexity.

I initially used **Stable Diffusion v1.5**, which showed promising results, but versions 2.0 and SDXL didn’t work as well for my needs. Then came the shift to `Flux-1.dev`, and VRAM issues quickly followed. Even my friend’s RTX 3090 with 24GB wasn’t enough!

I solved this by:

- **Precomputing prompt embeddings**, skipping the text encoder and tokenizer.
- Using `torch.float16` precision.
- Employing a **4-bit transformer**.

Finally, everything ran smoothly, and that felt like a major victory.

---

## Entropy Calculation and Image Selection

Now we’re getting into the more sophisticated stuff. To manipulate an image effectively, understanding its complexity is crucial. Calculating **entropy** helps us get a sense of how aggressive we can be with transformations.

### What Is Entropy?

**Entropy** measures uncertainty or randomness. In our case, it’s a way to gauge an image's complexity. This understanding lets us adjust our parameters to make sure we’re preserving the visual quality of the image.

### Calculating Entropy

Here's a simple code snippet I used for entropy calculation:

```python
import numpy as np
from PIL import Image

def calculate_entropy(image):
    grayscale_image = image.convert("L")
    pixel_data = np.array(grayscale_image)
    histogram = np.histogram(pixel_data, bins=256, range=(0, 255))[0]
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
```

### Why Entropy Matters

- **High Entropy Images**: Images with high entropy have a lot of detail, and manipulating these requires extra caution to avoid significant quality degradation. Small transformations can lead to noticeable changes, so it’s important to carefully balance each adjustment.
  
- **Low Entropy Images**: These tend to retain watermarks even after attacks, especially in smooth regions. The reasoning here is that smoother areas are more likely to hold on to watermark patterns because they lack the visual chaos present in more detailed images.
  
- **How I Used It**: Using the 0.1 and 0.9 quantiles, I mapped entropy values to a scale between `0` and `1`, with `1` representing the lowest entropy. This allowed me to better decide the strength of the diffusion—low entropy images could tolerate more aggressive manipulation without losing too much of their core features.

---

## Edge Detection with Canny Edge Detector

To prevent losing essential structural details while manipulating images, I employed the **Canny Edge Detector** to generate edge maps. These edge maps were key in maintaining the composition while tweaking other attributes.

### Generating Edge Maps

The Canny Edge Detector helped maintain key features throughout the manipulation process. Here’s how I generated the edge maps:

```python
import cv2

canny_tuple = [(100, 200)]
cannies = []

for low_threshold, high_threshold in canny_tuple:
    canny = cv2.Canny(image_resized_1024_np, low_threshold, high_threshold)
    canny = np.stack([canny]*3, axis=-1)  # Maintain consistent 3-channel format
    canny_resized_1024 = resize_image(Image.fromarray(canny), 1024)
    canny = Image.fromarray(np.array(canny_resized_1024))
    cannies.append(canny)
```

### Choosing Parameters

I used thresholds of `low: 100` and `high: 200` for the Canny edge detection. These are commonly used defaults and produced satisfactory results during visual inspection. It’s always good to visually inspect edge maps since edge detection is inherently subjective.

---

## Image Resizing Strategy

During the attack phase, I resized the images from `512x512` to `1024x1024` and then reverted them back to `512x512` after processing.

### Why Resize?

Initially, performing the Img2Img purification at `512x512` led to visible artifacts. Upscaling to `1024x1024` provided more room for the model to operate, leading to better visual quality. After the attack, I reverted the images back to their original resolution.

Even though I didn’t do a quantitative analysis, the visual quality definitely improved. I used the **LANCZOS** method for resizing, which is known for producing sharp results during both upscaling and downscaling.

---

## Purification with Varying Parameters

Now comes the exciting part—executing the actual attack. This stage is where I adjusted various parameters to generate different versions of the manipulated images, each with varying levels of transformation.

### Hyperparameters in Play

- **Guidance Values**: `[1, 2, 3, 4, 5, 6, 7, 8]` were tested to determine how strictly the model should follow its guidance during transformation.
  
- **Prompt Embeds**: To manage GPU constraints, the prompt embeds were kept generic for all images. The prompt I used was: `"best quality, no artifacts, extremely detailed"`. Embedding this prompt beforehand allowed me to bypass using heavy text encoders during the attack.
  
- **Annotations**: I manually annotated approximately 60 images with visible perturbations as `high_perturbation` and the rest as `low_perturbation`. My speculation is that these 60 images are primarily StegaStamp watermarks.

### Setting Strength Values

- **Low Perturbation Budget**: Strength between `0.3` to `0.4` was used to balance perturbation with visual quality preservation.
  
- **High Perturbation Budget**: Strength between `0.55` to `0.65` was used when I needed more significant changes without compromising the natural look of the images.

### The Code for Execution

```python
for canny in cannies:
    for guidance in guidances:
        for count in range(3):
            generated_images.extend(
                pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    image=image_resized_1024,
                    num_inference_steps=20,
                    control_image=canny,
                    controlnet_conditioning_scale=0.7,
                    height=1024,
                    width=1024,
                    strength=(0.3 if low_perturbed else 0.55) + entropy_strength,
                    guidance_scale=guidance,
                ).images
            )
```

#### Why Multiple Iterations?

Running the pipeline with varying combinations of guidance and seeds allowed me to produce a range of results. Having a diverse set of outputs enabled me to select images that best maintained the original image quality.

#### Personal Comment

I believe a higher strength value, such as `0.8`, still produces high-quality images with significant similarity to the source image, especially with the support of ControlNet. Additionally, it is more likely to remove watermarks. However, it’s clear that selecting the lowest strength for the diffusion process is essential for achieving the best results in quality metrics.

---

## Post-Processing with PairOptimizer

After generating the attacked images, I used a custom tool called **PairOptimizer** to bring the manipulated images closer to their original versions in terms of visual quality.

### How Does PairOptimizer Work?

PairOptimizer makes fine-tuned adjustments (like hue, gamma, brightness, etc.) iteratively over multiple passes—**50 iterations** in my case.

Here's an overview of how it works:

```python
adjustments = [
    ImageAdjustment('exposure', adjust_exposure, 0.0, -0.5, 0.5),
    ImageAdjustment('gamma', adjust_gamma, 1.0, 0.7, 1.3),
    ImageAdjustment('brightness', adjust_brightness, 1.0, 0.5, 1.5),
    ImageAdjustment('contrast', adjust_contrast, 1.0, 0.5, 1.5),
    ImageAdjustment('saturation', adjust_saturation, 1.0, 0.5, 1.5),
    ImageAdjustment('hue', adjust_hue, 0.0, -0.2, 0.2),
    ImageAdjustment('temperature', adjust_temperature, 0.0, -2.5, 2.5),
    ImageAdjustment('tint', adjust_tint, 0.0, -2.5, 2.5),
    ExtendedColorMixerAdjustment(num_hues=10, num_sats=4, num_vals=4)
]
```

#### ExtendedColorMixerAdjustment

This adjustment makes masks based on (hue, saturation, value) ranges and learns parameters for each mask individually. Gaussian kernels were used for masking, and parameter ranges were kept limited to avoid introducing artifacts.

### Insights from Metrics

I aimed to minimize the **PSNR** (Peak Signal-to-Noise Ratio) and **MSSSIM** (Multi-Scale Structural Similarity) loss. These metrics were key in ensuring the attacked images were still visually close to the originals.

#### Why This Matters

During the diffusion process, the colors and other properties of an image can shift slightly, which affects its quality metrics. Post-processing with PairOptimizer mitigates these changes and brings the manipulated image closer to the original.

I got this idea from my photography experience with Lightroom—using the color mixer adjustments in Lightroom inspired me to try a similar approach programmatically.

*I think one advantage of this pipeline is that it can almost be adapted to any already existing attack method to enhance visual similarity.*

![Pair Optimizer](PairOptmizer.png)


---

## Optional Extra Purification

Once I had a set of optimized images, I repeated steps 5 and 6 with much lower strength values (`0.05`, `0.1`). This extra purification step was faster and helped me further refine the images while ensuring **erasing the invisible** remained effective.

- **Why Repeat?** With a lower strength, the purification process is significantly faster, allowing us to explore a wider range of hyperparameters and identify an image with the lowest loss value on specified metrics.

- **No Need for ControlNet**: Since the strength is low, we no longer need to use ControlNet. This makes the pipeline even lighter and faster.

---

## Handling TreeRing Attacks

**TreeRing Attacks** were particularly challenging due to their latent nature (I learned about them in the parallel BeigeBox track). These attacks hide within image features, making them difficult to detect and remove.

### Solution: Crop and Rotate

After examining numerous images affected by TreeRing attacks, I manually annotated about 70 images suspected of having these compositional unpleasant structures. I found that applying a **`0.98` crop combined with a `3-degree` rotation** was highly effective at removing most hidden messages while retaining the images' visual quality.

#### Why Crop and Rotate?

Cropping and rotating the image disrupts the exact placement of latent features and embedded frequencies, reducing the presence of TreeRing watermarks. The idea was that even minor alterations in the spatial domain could break the precise positioning needed for these attacks to remain intact.

### How Did I Arrive at This?

While working on the parallel BeigeBox track, I found that a `0.90` crop almost entirely destroyed TreeRing attacks, but at the cost of substantial quality loss. To balance quality retention and effectiveness, I tested different crop ratios and slight rotations, which led me to the final solution.

---

## Final Quality Improvements Using Frequency-Based Adjustments

To finalize the image manipulations, I applied a **frequency-based enhancement** using Fast Fourier Transform (FFT). This was inspired by a project I did a few years ago called **Hybrid Images** that combines two images based on their frequencies.

### How I Did It

I replaced the first **`3` frequencies** of the attacked image with those from the original image. These frequencies represent the overall structure and broad visual tones of the image.

- **Result**: By using just the first three frequencies from the original image, I was able to bring the manipulated images closer to their original versions in terms of overall look and feel, without sacrificing the effectiveness of the attack.

---

## What Didn’t Work

### Surrogate Adversarial Attacks

I initially focused on surrogate attacks using the WAVES benchmark surrogates and models from the paper *"Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks"*. Even with a high epsilon (`24/255`) and a large number of steps (`1000`) in the PGD attack, these models were unsuccessful.

- **Diffusion Pipeline Integration**: I tried integrating surrogates into the image diffusion pipeline, even adding them into the guidance loss function. Despite my best efforts, I observed no noticeable improvements.

### Rinsing

This method is effective at removing pixel and image-level watermarks and might even succeed in removing some TreeRing watermarks. However, I found the quality degradation to be unjustifiable.

### Pattern Removal Using Average Differences

Towards the end of my experiments, I came across an interesting paper: [Average Difference in Watermarks](https://arxiv.org/pdf/2406.09026). Since I had created TreeRing and StegaStamp versions of `10,000` images, I calculated the average image difference across all pairs and saw intriguing patterns for each watermarking algorithm. However, adding or subtracting these patterns—whether on large or small scales—did not work effectively.

### Other Transformations for TreeRing Attacks

Since crop and resize worked surprisingly well on these types of attacks, I experimented with other transformations ranging in degrees of freedom, including Translation, Affine, and Homography. Additionally, I explored optical distortions and a method I coined "patch-aware stretching" (I think I invented it, lol). However, none of these methods performed as effectively as the simple crop and scale approach.

### Reverse Resizing for TreeRing Attacks

Reflecting on the vulnerability of TreeRing watermarks, I considered an alternative approach: instead of cropping the central `0.9` and rescaling, I scaled down the entire image and outpainted the margins to fill a `512x512` canvas. Using the FluxInpaint pipeline, this method showed decent results but performed on par with the initial approach. The key drawback was its computational cost—about 1000 times heavier than simple crop and scale—and the complexity of parameter tuning for acceptable outpainting. Despite these challenges, it has a notable advantage: it ensures no significant part of the image is removed during processing.


#### StegaStamp Average Pattern
![StegaStamp](stega_difference.png)

#### TreeRing Average Pattern
![TreeRing](tr_difference.png)

---

## My Alternate Approach

Before using Flux, my primary pipeline consisted of a vanilla text-to-image diffusion model with Canny ControlNet. The overall strategy was similar, but instead of setting specific strengths on the Img2Img pipeline, I used **IP Adapters**, which essentially serve as image prompts.

By combining the Canny image with the image prompt, I guided text-to-image diffusion models to produce better results. The results were promising for watermark removal while maintaining image quality, composition, and appearance. Unfortunately, IP Adapters were not yet implemented in the HuggingFace library, which forced me to switch to using Img2Img to simulate the effect of image prompts.

---

## Fun Facts!

- Over the last month of the competition, I spent about 5 to 10 hours each day coding, looking at images, and brainstorming ideas. It was intense, but also very rewarding.

- One of the hardest parts for me was dealing with slow internet in my country. Every time I wanted to upload a submission, I had to download it from Google Colab or Kaggle to my computer and then re-upload it. This took around 20 minutes each time and ended up being the most internet I’ve ever used in a month!

- To be honest, I joined the competition mainly for the prize money because I wanted to buy a camera (I love photography!). But along the way, I ended up learning so much about diffusion models, samplers, schedulers, and tools like ControlNet, IP Adapters, text-to-image, image-to-image, and inpainting. By the end, I felt like the knowledge I gained was even more valuable than the prize. It’s something that will definitely help me in my research in the future.

- I honestly didn’t expect the watermarks to be so strong. My initial thought was to focus mostly on quality with just a little purification. For one of my first submissions, I concentrated heavily on making the images look great and visually ensuring there weren’t any obvious patterns. I was pretty confident about that submission, but it ended up being one of my worst, with a watermark performance score of `0.96`.  This experience has made me really interested in exploring this area further. It’s closely connected to my current research in trustworthy machine learning, and I’d love to learn more about how to make models better at handling challenges like this in the future.

---

## Acknowledgments

I want to take a moment to thank the incredible organizers and everyone who made this competition possible. A special thanks to Professor Furong Huang and her team at the University of Maryland, along with Bang An, Chenghao Deng, and Mucong Ding, for their dedication and support throughout the event. Their guidance and assistance were invaluable, and I’m truly grateful for their efforts in making this experience so enriching and rewarding!

---

## References

- [Flux Pipeline Documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
- [ControlNet Pipeline Documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet)
- [WAVES Benchmark](https://arxiv.org/abs/2401.08573)
- [Robustness of AI-Image Detectors](https://arxiv.org/abs/2310.00076)
- [Average Difference in Watermarks](https://arxiv.org/pdf/2406.09026)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Hybrid Images](https://en.wikipedia.org/wiki/Hybrid_image)