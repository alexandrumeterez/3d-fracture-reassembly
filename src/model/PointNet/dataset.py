class dataset_pc(Dataset):
    
    def __init__(self, object_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.triplets_list = []
        self.kp_files = []
        self.anc = np.zeros((1,128))
        self.pos = np.zeros((1,128))
        self.neg = np.zeros((1,128))
        object_type =os.listdir(object_path)
        for object_t in object_type:
            # print(object_t)
            self.object_pc_dir = os.path.join(object_path,object_t)
            # objects = os.listdir(object_t_dir)
            # for object_ in objects:
            #     obj_pc_dir = os.path.join(object_t_dir,object_)

            triplet_path = os.path.join(self.object_pc_dir,'triplets')
            files = sorted(glob.glob(triplet_path+"/*.npz"))
            self.triplets_list.extend(files)
        for obj_file in self.triplets_list:
            dict = np.load(obj_file)
            anc = dict['anchor']
            pos = dict['positive']
            neg = dict['negative']
            self.anc = np.append(self.anc, anc, axis=0)
            self.pos = np.append(self.pos, pos, axis=0)
            self.neg = np.append(self.neg, neg, axis=0)
        self.anc = np.delete(self.anc, 0, axis = 0)
        self.pos = np.delete(self.pos, 0, axis = 0)
        self.neg = np.delete(self.neg, 0, axis = 0)
       


        # anchor, positive, negative = load_data(triplet_files)

    def __len__(self):
        # print(len(self.triplets_list))

        return self.anc.shape[0]

    # def load_data(self,triplet_files):
        
    #     anchor=np.zeros((1,128))
    #     positive=np.zeros((1,128))
    #     negative=np.zeros((1,128))

    #     for obj_file_list in triplet_files:
    #         # for f in obj_file_list:
    #         dict = np.load(obj_file_list)
    #         anc = dict['anchor']
    #         pos = dict['positive']
    #         neg = dict['negative']

    #         anchor = np.append(anchor, anc, axis=0)
    #         positive = np.append(positive, pos, axis=0)
    #         negative = np.append(negative, neg, axis=0)
    #             # print(f)
    #     anchor = np.delete(anchor, 0, axis = 0)
    #     positive = np.delete(positive, 0, axis = 0)
    #     negative = np.delete(negative, 0, axis = 0)
    #     # pdb.set_trace()

    #     return anchor, positive, negative

    def __getitem__(self, idx):

        data = {'anchor': self.anc[idx], 'positive': self.pos[idx], 'negative': self.neg[idx]}
        return data
        
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
