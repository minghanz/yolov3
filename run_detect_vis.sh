#!/bin/sh
# ############## lturn
# # weather="sun_long"
# # video_name="2020-07-14 00-49-21_1hr"

# weather="night_long"
# video_name="2020-07-14 20-50-17_1hr"
# calib_name="lturn"
# weight="best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout"
# python detect_vis_on_ori.py \
# --path_result "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/${weight}/${weather}/bev_${video_name}.txt" \
# --path_video_bev "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/${weather}/bev_${video_name}.mkv" \
# --path_video_ori "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/${weather}/ori_${video_name}.mkv" \
# --path_video_bev_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/bev_${video_name}.mkv" \
# --path_video_ori_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/ori_${video_name}.mkv" \
# --calib ${calib_name} --bspec ${calib_name} #--frame_start 14400

############## roundabout
# weather="rain_large"
# video_name="_30000"
# calib_name="roundabout"

# weather="night_large"
# video_name="_10min_4x"
# calib_name="roundabout"

weather="snow_large"
video_name="_2020-12-17 13-40-59_1hr"
calib_name="roundabout"

weight="best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout"
python detect_vis_on_ori.py \
--path_result "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/outputs/${weight}/${weather}/bev${video_name}.txt" \
--path_video_bev "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/videos/${weather}/bev${video_name}.mkv" \
--path_video_ori "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/videos/${weather}/ori${video_name}.mkv" \
--path_video_bev_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/bev${video_name}.mkv" \
--path_video_ori_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/ori${video_name}.mkv" \
--calib ${calib_name} --bspec ${calib_name} --frame_start 0

# ############### KoPER
# weather="bw"
# video_name="Sequence1c"
# calib_name="KoPER"
# bspec_num=1
# weight="best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout"

# python detect_vis_on_ori.py \
# --path_result "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/${video_name}/KAB_SK_${bspec_num}_undist/outputs/${weight}/bev.txt" \
# --path_video_bev "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/${video_name}/KAB_SK_${bspec_num}_undist/videos/bev.avi" \
# --path_video_ori "/media/sda1/datasets/extracted/roadmanship_format/KoPER_onetake/${video_name}/KAB_SK_${bspec_num}_undist/videos/ori.avi" \
# --path_video_bev_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/bev_${video_name}.mkv" \
# --path_video_ori_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/ori_${video_name}.mkv" \
# --calib ${calib_name} --bspec ${calib_name} --frame_start 0 --bspec_num ${bspec_num}


# weather="normal"
# video_name="4_right"
# calib_name="BrnoCompSpeed"
# weight="best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedqs_strip_giout"
# ################ BrnoCompSpeed
# python detect_vis_on_ori.py \
# --path_result "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name}/outputs/${weight}/bev.txt" \
# --path_video_bev "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name}/videos/bev.avi" \
# --path_video_ori "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name}/videos/ori.avi" \
# --path_video_bev_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/bev_${video_name}.mkv" \
# --path_video_ori_out "/media/sda1/datasets/extracted/roadmanship_format/for_paper/${weather}/ori_${video_name}.mkv" \
# --calib_path "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/session${video_name}/calibs/system_dubska_optimal_calib.json" \
# --calib ${calib_name} --bspec ${calib_name} --frame_start 0 --bspec_num 4.3 --calib_new_u 852 --calib_new_v 480


### want examples: snow, rain, shadow, roundabout, KOPER, BrnoCompSpeed

# parser.add_argument("--path_result", type=str)
# parser.add_argument("--path_video_bev", type=str)
# parser.add_argument("--path_video_ori", type=str)
# parser.add_argument("--path_video_bev_out", type=str)
# parser.add_argument("--path_video_ori_out", type=str)
# parser.add_argument("--calib", type=str)
# parser.add_argument("--bspec", type=str)
# parser.add_argument("--calib_path", type=str, required=False, default=None)
# parser.add_argument("--bspec_num", type=float, required=False, default=None)
# parser.add_argument("--calib_new_u", type=int, required=False, default=None)
# parser.add_argument("--calib_new_v", type=int, required=False, default=None)
# parser.add_argument("--dry_run", action="store_true")
