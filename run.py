import os 
import os.path as osp 
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_video_path", type=str)
    parser.add_argument("--input_audio_path", type=str) 
    args = parser.parse_args()

    video_file = args.input_video_path.split('/')[-1]
    video_name = video_file[:-4] # remove ext .mp4 
    audio_file = args.input_audio_path.split('/')[-1]
    audio_name = audio_file[:-4] # remove ext .wav 

    base = os.getcwd()

    # 1. Resize the input video to 320p, and generate the intra-frame coding version of the input video to make the segmented movie shots more accurate. 

    infra_video_file = video_name + '-infra.mp4' 
    infra_video_path = osp.join(base, infra_video_file) 
    cmd = f'python ./utils/infra_video_ffmpeg.py --input_video_path {args.input_video_path} --output_video_path {infra_video_path}' 
    os.system(cmd)  

    # rename the input video to '{video_name}-raw.mp4'
    new_input_video_path = osp.join(base, video_name+'-raw.mp4')
    cmd = f'mv {args.input_video_path} {new_input_video_path}'

    # the rescaled 320p video will be renamed to '{video_name}.mp4'
    rescaled_video_file = video_name + '.mp4' 
    rescaled_video_path = osp.join(base, rescaled_video_file) 
    cmd = f'python ./utils/rescale_movies_ffmpeg.py --input_video_path {args.input_video_path} --output_video_path {rescaled_video_path}' 
    os.system(cmd) 

    # 2. Segment the input 320p video into movie shots through BaSSL. 
    #   By default, the scene_seg_info will be saved at './{video_name}.json'
    #   the segmented video shots will be saved at './results/shot_split_video/{video_name}'
    cmd = f'python ./segmentation/scene_segmentation_bassl.py --input_video_path {rescaled_video_path}' 
    os.system(cmd) 

    # 3. Segment the input music into music shots through ruptures. 
    #   By default, the audio_seg_info will be saved at './{audio_name}.json'
    #   the segmented audio shots will be saved at './seg_audio_shots/{audio_name}' 
    cmd = f'python ./segmentation/audio_segmentation_ruptures.py --input_audio_path {args.input_audio_path}' 
    os.system(cmd) 

    # Note: the (4) and (5) steps, the python files should be placed at the ImageBind repo (https://github.com/facebookresearch/ImageBind), e.g., at './ImageBind/' directory. 
    # 4. Encode the movie shots into shot-level visual embeddings through ImageBind. 
    #   By default, the extracted visual embs will be saved at './video_embs/{video_name}.npy' 
    save_shot_base = osp.join('./results/shot_split_video', video_name) 
    save_video_embs_dir = osp.join(base, 'video_embs') 
    cmd = f'python ./ImageBind/extract_video_embs.py --save_shot_base {save_shot_base} --save_video_embs_dir {save_video_embs_dir}'
    os.system(cmd) 

    # 5. Encode the music shots into shot-level acoustic embeddings through ImageBind. 
    #   By default, the extracted visual embs will be saved at './audio_embs/{audio_name}.npy' 
    save_bar_base = osp.join('./seg_audio_shots', audio_name) 
    save_audio_embs_dir = osp.join(base, 'audio_embs')
    cmd = f'python ./ImageBind/extract_audio_embs.py --save_bar_base {save_bar_base} --save_audio_embs_dir {save_audio_embs_dir}'
    os.system(cmd) 

    # 6. With the processed embeddings, we can just run  python trailer_generator.py to generate the personalized trailers.
    audio_seg_info = osp.join(base, f'{audio_name}.json')
    video_seg_info = osp.join(base, f'{video_name}.json')
    cmd = f'python trailer_generator.py --video_name {video_name} --audio_name {audio_name} --movie_shot_info_path {video_seg_info} --audio_bar_info_path {audio_seg_info}'
    
