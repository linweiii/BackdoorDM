# ImageFix
python defense/model_level/t2ishield/t2ishield.py --backdoor_method villandiffusion_cond  --device cuda:0

# ObjectRep
python defense/model_level/t2ishield/t2ishield.py --backdoor_method badt2i_object  --device cuda:0 
python defense/model_level/t2ishield/t2ishield.py --backdoor_method eviledit  --device cuda:0 
python defense/model_level/t2ishield/t2ishield.py --backdoor_method paas_db  --device cuda:0 
python defense/model_level/t2ishield/t2ishield.py --backdoor_method paas_ti  --device cuda:0 
python defense/model_level/t2ishield/t2ishield.py --backdoor_method rickrolling_TPA  --device cuda:0

# ImagePatch
python defense/model_level/t2ishield/t2ishield.py --backdoor_method badt2i_pixel  --device cuda:0

# StyleAdd
python defense/model_level/t2ishield/t2ishield.py --backdoor_method badt2i_style  --device cuda:0
python defense/model_level/t2ishield/t2ishield.py --backdoor_method rickrolling_TAA  --device cuda:0
