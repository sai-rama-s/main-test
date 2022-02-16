PARAMS = {
    "inputs": {
        "optical_path": "./Mosaic_Output/RGB.tif",
        "sar_path": "./Mosaic_Output/SAR.tif",
        "noptical_path" :"./Mosaic_Output/RGB.tif"
        #"noptical_path" :"./Mosaic_Output/nOptical.tif"
    },
    "outputs": {
        "output_path": "./Outputs/",
        "order_id":    "./Coreg/",
        "optical_path": "./Coreg/Optical/",
        "sar_path": "./Coreg/Sar/",
        "s3": True,
        "s3_bucket": 'airborne-data',
        "s3_path": "sentinel/orders/"
    },
    
    "options": {
        "DEFAULT_OPTIONS_COMMON": {
            "verbose_main": True,
            "verbose": True,
            "plot": False,
            "save": False,
            "dir": "Common/",
            "path": "{}.jpg"
        },
        "DEFAULT_OPTIONS_OPTICAL": {
            "verbose_main": True,
            "verbose": True,
            "plot": False,
            "save": False,
            "dir": "Optical/",
            "path": "{}.jpg"
        },
        "DEFAULT_OPTIONS_SAR": {
            "verbose_main": True,
            "verbose": True,
            "plot": False,
            "save": False,
            "dir": "SAR/",
            "path": "{}.jpg"
        }
    },

    "preprocessing": {
        "crop": {
            "crop_start": [0, 0],
            "crop_size": [600, 600]
        },
        "normalise": {
            # Choose between
            # DEFAULT: MinMax (Optical), ReScale 5x (SAR)
            # [MinMax, ReScale, Histogram_Norm, Histogram_Eq, Adaptive_Histogram, MinMax_GaussCorrection, Digitize, Histogram_Matching]
            "optical": {
                "func": "MinMax", 
                "params": {
                    # Adaptive Histogram
                    "clip_limit": 0.01,

                    # MinMax with Gaussian Correction
                    "c": 1,
                    "gamma": 0.6,

                    # ReScale
                    "minVal": 0.0,
                    "maxVal": 2.0
                }
            },
            "sar": {
                "func": "MinMax",
                "params": {
                    # Adaptive Histogram
                    "clip_limit": 0.01,

                    # MinMax with Gaussian Correction
                    "c": 1,
                    "gamma": 0.6,
                    
                    # ReScale
                    "minVal": 0.0,
                    "maxVal": 2.0
                }
            }
        },
        "denoise": {
            "optical": {},
            "sar": {
                "func": "BM3D",
                "params": {
                    "sigma_psd": 20.0/255.0
                }
            }
        },
        "denoised_normalise": {
            "func": "Histogram_Matching",
            "params": {
                # Adaptive Histogram
                "clip_limit": 0.01,

                # MinMax with Gaussian Correction
                "c": 1,
                "gamma": 0.6,
                
                # ReScale
                "minVal": 0.0,
                "maxVal": 5.0
            }
        }
    },

    "params": {
        "scale_space": {
            "diffusion": "g3",
            "sigma": 2.0,
            "S": 5,
            "ratio": (2**(1/5)),
            "d": 0.04,
            "GKernelSize": (3, 3),
            "use_harris": True
        },
        "keypoint_detection": {
            "optical": {
                "N_keypoints": 30,
                "thresholds": {
                    "log_threshold": -1.0
                },
                "windows": {
                    "extrema_window": 3,
                    "common_check_window": 3,
                    "overall_check_window": 1
                },
                "other": {
                    "ignores": [False, False],
                    "combine_weights": [0.5, 0.5]
                },
                "min_scales_detection": 2,
                "combined_method": False,
                "adaptive_thresholding": {
                    "N": 10,
                    "recursions": 3
                }
            },
            "sar": {
                "N_keypoints": 30,
                "thresholds": {
                    "log_threshold": -1.0
                },
                "windows": {
                    "extrema_window": 3,
                    "common_check_window": 3,
                    "overall_check_window": 1
                },
                "other": {
                    "ignores": [False, False],
                    "combine_weights": [0.5, 0.5]
                },
                "min_scales_detection": 2,
                "combined_method": False,
                "adaptive_thresholding": {
                    "N": 10,
                    "recursions": 3
                }
            },
            "repeatabilityDistThreshold": 1.5
        },

        "descriptor_generation": {
            "dense": {
                "optical": False,
                "sar": True
            },
            "descriptor": {
                # Choose between
                # DEFAULT: LogGabor
                # [LogGabor, GLOH, CFOG, Gabor]
                "func": "LogGabor",
                "params": {
                    # LogGabor
                    "LOGGABOR_sigma_0": 2.0,
                    "LOGGABOR_ratio": (2**(1/5)),
                    "LOGGABOR_N_RADIAL_BINS": 3,
                    "LOGGABOR_N_ANGULAR_BINS": 8,
                    "LOGGABOR_N_SCALES": 4,
                    "LOGGABOR_N_ORIENTATIONS": 6,
                    "LOGGABOR_kernel_size": (25, 25), # NOT USED
                    "LOGGABOR_keypoint_centric": False,
                    "LOGGABOR_binning": False,
                    "LOGGABOR_visualise": False,

                    # GLOH
                    "GLOH_sigma_0": 2.0,
                    "GLOH_ratio": (2**(1/5)),
                    "GLOH_N_RADIAL_BINS": 3,
                    "GLOH_N_ANGULAR_BINS": 8,

                    # CFOG
                    "CFOG_sigma": 2.0,
                    "CFOG_ksize": 3,
                    "CFOG_N_ORIENTATIONS": 8,
                    "CFOG_visualise": False,

                    # Gabor
                    "Gabor_Sigmas": (1, 5),
                    "Gabor_N_Orientations": 6,
                    "Gabor_Frequencies": (0.05, 0.25),
                    "Gabor_visualise": False
                }
            },
            "norm": False,
            "includeDescriptor": True,
            "includePosition": False,
            "positionMultiplier": 1.0
        },

        "match": {
            "func": "Template",
            "params": {
                "crossCheck": True,

                "DistFunc": "L2",

                # Search Window for matching
                "search_window": 10,

                "MatchType": "BestAvailable",
                "RANSAC": True
            }
        },

        "shift": {
            "func": "HistogramShift_TopKDirections",
            "params": {
                "HistogramShift_TopKDirections": {
                    "N_directions": 4,
                    "K": 1,
                    "topKHist": "weighted",
                    # transformMethod : Can be "weighted-mean" or "affine"
                    # "weighted-mean" -> predicts only translation
                    # "affine" -> predicts translation and rotation
                    "transformMethod": "weighted-mean"
                }
            }
        },

        "final_display": {
            "resolution": 4
        }
    }
}