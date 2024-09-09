import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


class BiosignalTableData:
    def __init__(
        self,
        batch_size,
        num_batches,
        classes,
        num_classes,
        view1_train,
        view1_eval,
        view2_train,
        view2_eval,
        num_samples,
        dim_samples,
        view1_modalities,
        view2_modalities,
    ):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.classes = classes
        self.num_classes = num_classes
        self.view1_train = view1_train
        self.view1_eval = view1_eval
        self.view2_train = view2_train
        self.view2_eval = view2_eval
        self.num_samples = num_samples
        self.dim_samples = dim_samples
        self.view1_modalities = view1_modalities
        self.view2_modalities = view2_modalities

        self.training_data = self.get_dataset(view1_train, view2_train)
        self.eval_data = self.get_dataset(view1_eval, view2_eval)

    @classmethod
    def generate(
        cls,
        window_length=60,
        classes=["Relax1", "PhysicalStress", "EmotionalStress", "CognitiveStress"],
        eval_subjects=["Subject17", "Subject18", "Subject19", "Subject20"],
    ):
        batch_size = 3000

        # Load data
        root_dir = "/volatile/datasets/biosignal/raw"
        if not os.path.exists(root_dir):
            root_dir = "/volatile/datasets/biosignal/raw"
            if not os.path.exists(root_dir):
                raise ValueError("Dataset not available!")

        subjects = ["Subject" + str(i) for i in range(1, 21)]
        subject_folders = [os.path.join(root_dir, subject) for subject in subjects]

        data_dict = {}
        for subject_folder in subject_folders:
            # Load and preprocess for every subject
            subject = subject_folder.split("/")[-1]

            # Load heartrate and SpO2 data
            hs_file = os.path.join(subject_folder, subject + "SpO2HR.csv")
            hs_df = pd.read_csv(hs_file)

            # Load Acc, EDA and Temperature data
            ate_file = os.path.join(subject_folder, subject + "AccTempEDA.csv")
            ate_df = pd.read_csv(ate_file)

            ate_second_values = ate_df["Second"].values
            ms_values = ate_second_values - ate_second_values.astype(int)
            ms_values_zero = ms_values == 0.0
            downsampled_ate_df = ate_df[ms_values_zero].reset_index(drop=True)

            assert len(downsampled_ate_df) == len(hs_df)
            assert np.all(
                downsampled_ate_df["Second"].values.astype(int)
                == hs_df["Second"].values
            )

            # Join dfs and drop duplicates
            joined_df = hs_df.join(downsampled_ate_df, rsuffix="r")
            assert (
                joined_df["Hour"].astype(float).equals(joined_df["Hourr"].astype(float))
            )
            assert (
                joined_df["Minute"]
                .astype(float)
                .equals(joined_df["Minuter"].astype(float))
            )
            assert (
                joined_df["Second"]
                .astype(float)
                .equals(joined_df["Secondr"].astype(float))
            )
            assert joined_df["Label"].equals(joined_df["Labelr"])
            joined_df = joined_df.drop(
                columns=["Hourr", "Minuter", "Secondr", "Labelr"]
            )
            joined_df = joined_df[
                [
                    "Hour",
                    "Minute",
                    "Second",
                    "HeartRate",
                    "SpO2",
                    "AccZ",
                    "AccY",
                    "AccX",
                    "Temp",
                    "EDA",
                    "Label",
                ]
            ]
            data_dict[subject] = joined_df

        subjects = list(data_dict.keys())
        labels = np.unique(data_dict["Subject1"]["Label"]).tolist()

        # Relabel data
        for subject in subjects:
            data_dict[subject]["NewLabel"] = ""
            label_changes = (
                np.where(
                    np.abs(np.diff(pd.factorize(data_dict[subject]["Label"])[0])) > 0
                )[0]
                + 1
            )
            label_changes = np.concatenate(
                [label_changes, np.asarray(len(data_dict[subject]["Label"]))[None,]]
            )

            last_label_change = 0
            for i, label_change in enumerate(label_changes):
                label_d = data_dict[subject][last_label_change:label_change]
                labels = np.unique(label_d["Label"])
                assert len(labels) == 1
                label = labels[0]

                if label == "Relax":
                    if i == 0:
                        new_label = "Relax1"
                    elif i == 2:
                        new_label = "Relax2"
                    elif i == 5:
                        new_label = "Relax3"
                    elif i == 7:
                        new_label = "Relax4"
                    else:
                        raise ValueError

                elif label == "PhysicalStress":
                    new_label = "PhysicalStress"

                elif label == "EmotionalStress":
                    if i == 3:
                        new_label = "CognitiveStress"
                    elif i == 6:
                        new_label = "EmotionalStress"

                elif label == "CognitiveStress":
                    new_label = "CognitiveStress"

                data_dict[subject].loc[
                    last_label_change:label_change, "NewLabel"
                ] = new_label
                last_label_change = label_change

        new_labels = np.unique(data_dict["Subject1"]["NewLabel"]).tolist()
        label_encoder = preprocessing.LabelEncoder()
        label_encoder = label_encoder.fit(classes)

        # Create time windows
        windows_dict = dict()
        modalities = ["HeartRate", "SpO2", "AccX", "AccY", "AccZ", "Temp", "EDA"]
        modalities_wo_acc = ["HeartRate", "SpO2", "Temp", "EDA"]
        meta_infos = ["NewLabel"]
        keys = modalities + meta_infos

        for subject in subjects:
            windows_dict[subject] = {key: list() for key in keys}
            subject_labels = data_dict[subject]["NewLabel"].unique().tolist()
            for label in subject_labels:
                tmp_df = data_dict[subject][data_dict[subject]["NewLabel"] == label]
                num_samples = len(tmp_df)
                num_windows = np.floor(num_samples / window_length).astype(int)
                for i in range(num_windows):
                    window = tmp_df[i * window_length : (i + 1) * window_length]
                    for key in keys:
                        windows_dict[subject][key].append(window[key].to_numpy())

            for key in keys:
                windows_dict[subject][key] = np.stack(
                    windows_dict[subject][key], axis=0
                )

        for subject in subjects:

            for modality in modalities_wo_acc:
                sensor_data = windows_dict[subject][modality]
                windows_dict[subject][modality] = {
                    "std": np.std(sensor_data, axis=1),
                    "mean": np.mean(sensor_data, axis=1),
                }

            for modality in ["AccX", "AccY", "AccZ"]:
                windows_dict[subject][modality] = {
                    "std": np.std(windows_dict[subject][modality], axis=1),
                }

        # Combine subjects
        train_subjects = [
            item for item in subjects if item not in eval_subjects
        ]  # +test_subjects]

        def combine_subjects(
            list_of_subjects, windows_dict, label_encoder, selected_classes
        ):

            view1_modalities = ["SpO2", "HeartRate"]
            view2_modalities = ["AccX", "AccY", "AccZ", "Temp", "EDA"]
            label = "NewLabel"

            view1_df = None
            view2_df = None

            for subject in list_of_subjects:
                subject_view1_df = None
                for modality in view1_modalities:
                    modality_df = pd.DataFrame(windows_dict[subject][modality])
                    modality_df.columns = [
                        modality + "_" + col for col in modality_df.columns
                    ]
                    if subject_view1_df is None:
                        subject_view1_df = modality_df
                    else:
                        subject_view1_df = subject_view1_df.join(modality_df)

                subject_view2_df = None
                for modality in view2_modalities:
                    modality_df = pd.DataFrame(windows_dict[subject][modality])
                    modality_df.columns = [
                        modality + "_" + col for col in modality_df.columns
                    ]
                    if subject_view2_df is None:
                        subject_view2_df = modality_df
                    else:
                        subject_view2_df = subject_view2_df.join(modality_df)

                labels = windows_dict[subject]["NewLabel"][:, 0]
                subject_view1_df["Labels"] = labels
                subject_view2_df["Labels"] = labels

                subject_view1_df["Subject"] = subject
                subject_view2_df["Subject"] = subject

                if view1_df is None:
                    view1_df = subject_view1_df
                    view2_df = subject_view2_df
                else:
                    view1_df = pd.concat([view1_df, subject_view1_df]).reset_index(
                        drop=True
                    )
                    view2_df = pd.concat([view2_df, subject_view2_df]).reset_index(
                        drop=True
                    )

            # Filter selected classes
            view1_filter = view1_df["Labels"].isin(selected_classes)
            view2_filter = view2_df["Labels"].isin(selected_classes)
            assert np.all(view1_filter == view2_filter)

            view1_df = view1_df[view1_filter]
            view2_df = view2_df[view2_filter]

            view1_subjects = view1_df.pop("Subject")
            view2_subjects = view2_df.pop("Subject")
            assert np.all(view1_subjects == view2_subjects)

            view1_labels = view1_df.pop("Labels")
            view2_labels = view2_df.pop("Labels")
            assert np.all(view1_labels == view2_labels)

            labels = view1_labels
            labels_int = label_encoder.transform(labels)

            view1_modalities = view1_df.columns.to_list()
            view2_modalities = view2_df.columns.to_list()

            view1_data = view1_df.to_numpy()
            view2_data = view2_df.to_numpy()

            view_1 = (view1_data, labels_int)
            view_2 = (view2_data, labels_int)
            return view_1, view_2, view1_modalities, view2_modalities

        view1_train, view2_train, view1_modalities, view2_modalities = combine_subjects(
            train_subjects,
            windows_dict=windows_dict,
            label_encoder=label_encoder,
            selected_classes=classes,
        )
        view1_eval, view2_eval, _, _ = combine_subjects(
            eval_subjects,
            windows_dict=windows_dict,
            label_encoder=label_encoder,
            selected_classes=classes,
        )

        assert np.all(view1_train[0].shape[0] == view2_train[0].shape[0])
        data_shape = view1_train[0].shape

        num_samples = data_shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        num_classes = len(classes)
        dim_samples = data_shape[1]

        return cls(
            batch_size=batch_size,
            num_batches=num_batches,
            classes=classes,
            num_classes=num_classes,
            view1_train=view1_train,
            view1_eval=view1_eval,
            view2_train=view2_train,
            view2_eval=view2_eval,
            num_samples=num_samples,
            dim_samples=dim_samples,
            view1_modalities=view1_modalities,
            view2_modalities=view2_modalities,
        )

    def get_dataset(self, view1, view2):
        assert np.all(view1[1] == view2[1])

        # Create dataset from numpy array in dict
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "nn_input_0": view1[0],
                "nn_input_1": view2[0],
                "labels": view1[1],
            }
        )

        # Batch
        dataset = dataset.batch(self.batch_size)

        dataset = dataset.cache()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
