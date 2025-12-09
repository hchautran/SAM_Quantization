def get_default_datasets():
    """Get default dataset configurations"""
    return [
        {
            "name": "DIS5K-VD",
            "im_dir": "./data/DIS5K/DIS-VD/im",
            "gt_dir": "./data/DIS5K/DIS-VD/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "DIS5K-TR",
            "im_dir": "./data/DIS5K/DIS-TR/im",
            "gt_dir": "./data/DIS5K/DIS-TR/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "ThinObject5k-TR",
            "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "DIS5K-TR",
            "im_dir": "./data/DIS5K/DIS-TR/im",
            "gt_dir": "./data/DIS5K/DIS-TR/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
            },
        
    ]