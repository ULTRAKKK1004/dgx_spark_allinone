import json

def get_stage1_image_gen_workflow(lecturer_img, prompt, job_id):
    """
    Stage 1: Generate professional background/lecturer using SDXL Image-to-Image + ReActor.
    Uses the uploaded image as base to preserve likeness.
    """
    lecture_prompt = f"A professional person in a suit, {prompt}, standing in a large modern auditorium, realistic lighting, front facing, looking at camera, masterpiece, 4k"
    
    return {
        "sdxl_loader": {
            "inputs": {
                "ckpt_name": "SDXL/sd_xl_base_1.0_0.9vae.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "load_ref": {
            "inputs": {
                "image": lecturer_img,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        "vae_encode": {
            "inputs": {
                "pixels": ["load_ref", 0],
                "vae": ["sdxl_loader", 2]
            },
            "class_type": "VAEEncode"
        },
        "pos_clip": {
            "inputs": {
                "text": lecture_prompt,
                "clip": ["sdxl_loader", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "neg_clip": {
            "inputs": {
                "text": "blurry, low quality, distorted face, bad anatomy, messy auditorium, glasses, sunglasses",
                "clip": ["sdxl_loader", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "sampler": {
            "inputs": {
                "seed": 42,
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 0.35,
                "model": ["sdxl_loader", 0],
                "positive": ["pos_clip", 0],
                "negative": ["neg_clip", 0],
                "latent_image": ["vae_encode", 0]
            },
            "class_type": "KSampler"
        },
        "decode": {
            "inputs": {
                "samples": ["sampler", 0],
                "vae": ["sdxl_loader", 2]
            },
            "class_type": "VAEDecode"
        },
        "reactor": {
            "inputs": {
                "enabled": True,
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "codeformer-v0.1.0.pth",
                "face_restore_visibility": 1,
                "codeformer_weight": 0.8,
                "detect_gender_source": "no",
                "detect_gender_input": "no",
                "source_faces_index": "0",
                "input_faces_index": "0",
                "console_log_level": 1,
                "input_image": ["decode", 0],
                "source_image": ["load_ref", 0]
            },
            "class_type": "ReActorFaceSwap"
        },
        "save_image": {
            "inputs": {
                "filename_prefix": f"stage1_{job_id}",
                "images": ["reactor", 0]
            },
            "class_type": "SaveImage"
        }
    }

def get_stage2_animatediff_workflow(ref_image, job_id):
    """
    Stage 2: Create a 4s idle loop using AnimateDiff (SDXL version).
    Reduced denoise to preserve the swapped face likeness.
    """
    return {
        "sdxl_loader": {
            "inputs": {
                "ckpt_name": "SDXL/sd_xl_base_1.0_0.9vae.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "ad_loader": {
            "inputs": {
                "model": ["sdxl_loader", 0],
                "model_name": "mm_sdxl_v10_beta.ckpt",
                "beta_schedule": "linear (AnimateDiff-SDXL)"
            },
            "class_type": "ADE_AnimateDiffLoaderWithContext"
        },
        "load_image": {
            "inputs": {
                "image": ref_image,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        "vae_encode": {
            "inputs": {
                "pixels": ["load_image", 0],
                "vae": ["sdxl_loader", 2]
            },
            "class_type": "VAEEncode"
        },
        "pos_clip": {
            "inputs": {
                "text": "subtle natural breathing, blinking, professional, looking at camera, high quality",
                "clip": ["sdxl_loader", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "neg_clip": {
            "inputs": {
                "text": "fast movement, chaotic, blurry, morphing, distorted face",
                "clip": ["sdxl_loader", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "sampler": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.35,
                "model": ["ad_loader", 0],
                "positive": ["pos_clip", 0],
                "negative": ["neg_clip", 0],
                "latent_image": ["vae_encode", 0]
            },
            "class_type": "KSampler"
        },
        "decode": {
            "inputs": {
                "samples": ["sampler", 0],
                "vae": ["sdxl_loader", 2]
            },
            "class_type": "VAEDecode"
        },
        "combine": {
            "inputs": {
                "images": ["decode", 0],
                "frame_rate": 8.0,
                "loop_count": 0,
                "filename_prefix": f"stage2_{job_id}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "save_output": True,
                "pingpong": False
            },
            "class_type": "VHS_VideoCombine"
        }
    }

def get_wan_s2v_workflow(ref_image, audio_file, prompt, job_id, chunk_idx, duration):
    """
    Wan2.2 S2V 14B High Quality Lip-Sync Workflow.
    Optimized for a professional lecturer in a hall.
    """
    # Calculate frames based on 16fps
    frame_length = int(duration * 16)
    if frame_length < 16: frame_length = 16 # min 1s
    if frame_length > 128: frame_length = 128 # limit to ~8s

    lecture_prompt = "A professional lecturer speaking confidently in a large hall, realistic lecture hall background, subtle natural gestures, front facing, masterpiece, high resolution, 4k"

    return {
        "unet": {
            "inputs": {
                "unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        "lora": {
            "inputs": {
                "model": ["unet", 0],
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                "strength_model": 1.0
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "model_sampling": {
            "inputs": {
                "model": ["lora", 0],
                "shift": 12.0,
                "shift_type": "default"
            },
            "class_type": "ModelSamplingSD3"
        },
        "vae": {
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "clip": {
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default"
            },
            "class_type": "CLIPLoader"
        },
        "audio_enc_loader": {
            "inputs": {
                "audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"
            },
            "class_type": "AudioEncoderLoader"
        },
        "load_image": {
            "inputs": {
                "image": ref_image,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        "load_audio": {
            "inputs": {
                "audio": audio_file
            },
            "class_type": "LoadAudio"
        },
        "audio_enc": {
            "inputs": {
                "audio_encoder": ["audio_enc_loader", 0],
                "audio": ["load_audio", 0]
            },
            "class_type": "AudioEncoderEncode"
        },
        "pos_clip": {
            "inputs": {
                "text": lecture_prompt,
                "clip": ["clip", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "neg_clip": {
            "inputs": {
                "text": "Vivid colors, overexposed, static, blurry details, worst quality, low quality, JPEG compression artifacts, ugly, mutilated, motionless image, cluttered background",
                "clip": ["clip", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "wan_s2v": {
            "inputs": {
                "positive": ["pos_clip", 0],
                "negative": ["neg_clip", 0],
                "vae": ["vae", 0],
                "audio_encoder_output": ["audio_enc", 0],
                "ref_image": ["load_image", 0],
                "width": 640,
                "height": 640,
                "length": frame_length,
                "batch_size": 1
            },
            "class_type": "WanSoundImageToVideo"
        },
        "sampler": {
            "inputs": {
                "model": ["model_sampling", 0],
                "positive": ["wan_s2v", 0],
                "negative": ["wan_s2v", 1],
                "latent_image": ["wan_s2v", 2],
                "seed": 42,
                "steps": 10,
                "cfg": 1.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0
            },
            "class_type": "KSampler"
        },
        "decode": {
            "inputs": {
                "samples": ["sampler", 0],
                "vae": ["vae", 0]
            },
            "class_type": "VAEDecode"
        },
        "combine": {
            "inputs": {
                "images": ["decode", 0],
                "audio": ["load_audio", 0],
                "frame_rate": 16.0,
                "loop_count": 0,
                "filename_prefix": f"chunk_{job_id}_{chunk_idx}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True
            },
            "class_type": "VHS_VideoCombine"
        }
    }