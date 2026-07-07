python /inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/script/make_jsonl.py -i /inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/output/musicgen_medium -o /inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/audiobox_aesthetics/input_musicgen_medium.jsonl

audio-aes /inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/audiobox_aesthetics/input_musicgen_medium.jsonl --batch-size 8 > /inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/audiobox_aesthetics/results_musicgen_medium.jsonl --ckpt /inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/audiobox_aesthetics/checkpoint.pt

