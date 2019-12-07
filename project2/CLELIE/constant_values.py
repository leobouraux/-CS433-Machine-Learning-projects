TRAINING_SIZE   = 20
SEED            = 1    # Set to None for random seed.
BATCH_SIZE      = 16  
NUM_EPOCHS      = 100
FG_THRESH       = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
IMG_PATCH_SIZE  = 16   # IMG_PATCH_SIZE should be a multiple of 4, image size should be an integer multiple of this number