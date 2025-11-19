# coding: utf-8


import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.p_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            ip_feat_file_path = os.path.join(dataset_path, config['vision_feature_p_file'])
            iz_feat_file_path = os.path.join(dataset_path, config['vision_feature_z_file'])
            tp_feat_file_path = os.path.join(dataset_path, config['text_feature_p_file'])
            tz_feat_file_path = os.path.join(dataset_path, config['text_feature_z_file'])
            vp_feat_file_path = os.path.join(dataset_path, config['video_feature_p_file'])
            vz_feat_file_path = os.path.join(dataset_path, config['video_feature_z_file'])
            if os.path.isfile(ip_feat_file_path):
                self.ip_feat = torch.from_numpy(np.load(ip_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(iz_feat_file_path):
                self.iz_feat = torch.from_numpy(np.load(iz_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(tp_feat_file_path):
                self.tp_feat = torch.from_numpy(np.load(tp_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(tz_feat_file_path):
                self.tz_feat = torch.from_numpy(np.load(tz_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(vp_feat_file_path):
                self.vp_feat = torch.from_numpy(np.load(vp_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(vz_feat_file_path):
                self.vz_feat = torch.from_numpy(np.load(vz_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)


