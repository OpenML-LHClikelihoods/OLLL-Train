import torch


class nLLsDataset(torch.utils.data.Dataset):
    def __init__(self, features, nLLs, dtype):
        self.features = [
            torch.tensor(features_onedataset, dtype=dtype)
            for features_onedataset in features
        ]
        self.nLLs = [
            torch.tensor(nLLs_onedataset, dtype=dtype)
            for nLLs_onedataset in nLLs
        ]

        # reduce the effectively used dataset to the length of the smallest dataset
        # (pure convenience, could use more data at the cost of more code)
        self.len = min(
            [len(features_onedataset) for features_onedataset in self.features]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return [
            (features[idx], nLLs[idx])
            for (features, nLLs) in zip(self.features, self.nLLs)
        ]