import torch
from transformers import CLIPImageProcessor, CLIPModel
from tqdm import tqdm

def compute_similairy(args, model, input_image,reference_image_feature,preprocess):
    # Load the two images and preprocess them for CLIP
    image_a = preprocess(input_image, return_tensors="pt")["pixel_values"].to(args.device)

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(reference_image_feature)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    
    sim = similarity_score.item()
    # print(sim)    
    return sim

def binary_search_helper(args, model, sd_pipeline, sample, threshold, reference_image_feature, processor):
    sample = sample.split(" ")
    if len(sample) == 1:
        return sample[0]

    middle = len(sample) // 2
    first_half = " ".join(sample[:middle])
    second_half = " ".join(sample[middle:])

    first_half_image = generate_with_seed(sd_pipeline, prompts=[first_half], seed=42,guidance=7.5)[0]
    first_half_ssim = compute_similairy(args, model, first_half_image, reference_image_feature, processor)

    if first_half_ssim > threshold:
        return binary_search_helper(args, model, sd_pipeline, first_half, threshold, reference_image_feature=reference_image_feature, processor=processor)
    else:
        second_half_image = generate_with_seed(sd_pipeline, prompts=[second_half], seed=42,guidance=7.5)[0]
        second_half_ssim = compute_similairy(args, model, second_half_image, reference_image_feature, processor)
        if second_half_ssim > threshold:
            return binary_search_helper(args, model, sd_pipeline, second_half, threshold, reference_image_feature=reference_image_feature, processor=processor)
        else:
            return None

def generate_with_seed(sd_pipeline, prompts, seed, guidance=7.5, save_image=False):
    '''
    generate an image through diffusers 
    '''
    outputs = []
    generator = torch.Generator("cuda").manual_seed(seed)

    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt=prompt,generator=generator,guidance_scale=guidance,num_inference_steps=25)['images'][0]
        # debug use
        image_name = f"./images/{prompt[:20]}.png"
        if save_image:
            image.save(image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs

def binary_search_trigger(args, model, sd_pipeline, samples, threshold, processor):
    triggers = []
    for sample in tqdm(samples, desc='Binary Searching'):
        sample = sample.strip()
        reference_image = generate_with_seed(sd_pipeline, prompts=[sample], seed=args.seed,guidance=7.5)[0]
        with torch.no_grad():
            image_features_reference = processor(reference_image, return_tensors="pt")["pixel_values"].to(args.device)
        trigger = binary_search_helper(args, model, sd_pipeline, sample, threshold, reference_image_feature=image_features_reference, processor=processor)
        if trigger == None:
            triggers.append('None')
        else:
            triggers.append(trigger)
    
    return triggers


def locate_clip(args, sd_pipeline_backdoor, backdoor_samples):
    model_ID = args.clip_model
    model = CLIPModel.from_pretrained(model_ID).to(args.device)
    processor = CLIPImageProcessor.from_pretrained(model_ID)

    threshold = args.locate_clip_threshold 

    triggers = binary_search_trigger(args, model, sd_pipeline_backdoor, backdoor_samples, threshold, processor)

    return triggers