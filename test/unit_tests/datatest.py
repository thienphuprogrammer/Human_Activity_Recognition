from src.data_pipeline.loaders.loaders import load_har_dataset

x_train, y_train = load_har_dataset('../../../data/raw/HAR/',
                                    '../../data/processed/HAR/',
                                    '../../../data/processed/HAR/UCF',
                                    max_dim=35, train_test='train')
