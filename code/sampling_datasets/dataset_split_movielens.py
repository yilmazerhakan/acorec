import torch
import random
import sys, os, builtins
import numpy as np
import os
import pandas as pd

current_directory = os.path.dirname(__file__)

# path for raw datasets and outputs
DS_PATH = os.path.join(current_directory, "../raw_datasets/")
DS_FOLDER = os.path.join(current_directory, "../datasets/")

# output dataset file prefix
DS_NAME = 'ml1m45'

# scenario
#  cs | lt
# cs : cold-start
# lt : long-tail
SCENARIO = "lt" 

# cross-validation fold count
CV_FOLD = 5

# reproduce seed code
SEED = 98765

# cold-start test user settings
heatUserThreshold = 20
TEST_USER_COUNT = 100

if SCENARIO=="lt":
    SEED = 56789
    TEST_USER_COUNT = 250

# we make differ seeds for cs and lt 
SEED += sum(ord(c) for c in SCENARIO)
random.seed(SEED)    
torch.manual_seed(SEED)


# Load Raw Data
data = []
with open(os.path.join(DS_PATH, "ml1m.dat"), "r") as f:    
    for line in f:
        line = line.strip().split(":")
        user_id = int(line[0])
        movie_id = int(line[1])
        rating = float(line[2])
        # timestamp = int(line[3])
        data.append([user_id, movie_id, rating])

data2 = pd.read_csv(os.path.join(DS_PATH, "ml1m.dat"),header=None, sep=':')
data = data2.values

# load Movie Labels for ML-1m
labels = []
with open(os.path.join(DS_PATH, "movies.dat"), "r") as f:
    for line in f:
        line = line.strip().split("::")
        movie_id = int(line[0])
        movie_name = line[1]
        genre = line[2]
        # timestamp = int(line[3])
        labels.append([movie_id, movie_name, genre])
labels = np.array(labels)

# We are numbering each ID uniquely to keep User and Movie IDs together.
unique_user_ids = np.unique(data[:, 0])
unique_movie_ids = np.unique(data[:, 1])
user_map = {id: i for i, id in enumerate(unique_user_ids)}
movie_map = {id: i for i, id in enumerate(unique_movie_ids)}

sorted_indices = np.argsort(unique_movie_ids[:])
item_dict = unique_movie_ids[sorted_indices][::1]

# 
user_indices = torch.LongTensor([user_map[id] for id in data[:, 0]])
movie_indices = torch.LongTensor([movie_map[id] for id in data[:, 1]])
indices = torch.stack([user_indices, movie_indices], dim=0)

values = torch.FloatTensor(data[:, 2])

# a sparse tensor to dense
size = (len(unique_user_ids), len(unique_movie_ids))
RR = torch.sparse.FloatTensor(indices, values, size)
RR_dense = RR.to_dense()
RR_dense[RR_dense < 4] = 0
RR_dense[RR_dense >= 4] = 1
RR_indices = RR_dense.nonzero().t()
RR_values = RR_dense[RR_indices[0], RR_indices[1]]
RR = torch.sparse.FloatTensor(RR_indices, RR_values, size)
RR = RR.to_dense()

# remove empty test users and items recursively
stop_clear = False
nU, nM = RR.shape
nU = int(nU)
nM = int(nM)
print(RR.shape)
max_iterations = 100
counter = 0

# remove empty test users and items recursively
# iterate until the condition is met
while not stop_clear and counter < max_iterations:
    # 
    row_sums = torch.sum(RR, dim=1)
    col_sums = torch.sum(RR, dim=0)

    # 
    row_mask = row_sums >= 1
    col_mask = col_sums >= 1

    rsum = int(row_mask.sum())
    csum = int(col_mask.sum())

    if rsum < nU:
        RR = RR[row_mask,:]

    if csum < nM:
        RR = RR[:,col_mask]
        del_labels = np.where(col_mask == False)[0]
        item_dict = np.delete(item_dict, del_labels)

    if (rsum+csum) == (nU+nM):
        stop_clear = True      

    nU, nM = RR.shape
    nU = int(nU)
    nM = int(nM)

    counter += 1

    print(RR.shape)


# create raw train set
Raw_Set = RR.clone()
nU, nM = Raw_Set.shape

# create probe set for selected fold
ProbeSetRand = torch.rand(nU, nM)
RawProbeSet = Raw_Set * ProbeSetRand

# Find Item Frequencies for Long-tail scenario
item_freqs = torch.norm(Raw_Set.float(), p=1, dim=0)
freq_vals, freq_ids = item_freqs.sort(descending=True, stable=True)
popular_click_threshold = torch.count_nonzero(Raw_Set) * (1/3)
total = 0
for last_item_indice in range(freq_vals.shape[0]):
    total += freq_vals[last_item_indice]
    # total arrived
    if total >= popular_click_threshold:
        break    

short_head_items =  freq_ids[:last_item_indice]
short_head_items = torch.tensor(short_head_items, dtype=torch.long)

for FOLD in range(1,CV_FOLD+1):

    Label_Set = item_dict

    generator = torch.Generator()

    # CV ranges in random 
    cval = 1 / CV_FOLD
    minr = (FOLD-1) * cval
    maxr = cval + minr

    print("# create probe set for selected fold")
    ProbeSet = RawProbeSet.clone()
    mask = torch.logical_or(ProbeSet < minr, ProbeSet >= maxr)
    ProbeSet[mask] = 0    
    ProbeSet[ProbeSet>0] = 1
    ProbeSet = ProbeSet.int()

    print("# remove probe set from train set")
    # remove probe set from raw set
    Train_Set = Raw_Set - ProbeSet

    nU, nM = Train_Set.shape

    print("# select candidate users")

    # select candidate users
    users_Ratings_Count_In_Train = torch.sum(Train_Set, dim=1)
    Candidate_Users_in_Train = torch.nonzero(users_Ratings_Count_In_Train >= heatUserThreshold)[:,0]
    
    if SCENARIO=="lt":
        ##  candidate users in the test set
        RA_temp=Train_Set.clone()
        RA_temp[:, short_head_items] = 0
        # candidate users must click at least a long -tail item in train set
        users_Ratings_Count_In_Train = torch.sum(RA_temp, dim=1)
        Candidate_Users_in_Train = torch.nonzero(users_Ratings_Count_In_Train > 0)[:,0]
        # remove short head items from Probe set for long-tail
        ProbeSet[:, short_head_items] = 0

    users_Ratings_Count_In_Probe = torch.sum(ProbeSet, dim=1)
    Candidate_Users_in_Test = torch.nonzero(users_Ratings_Count_In_Probe>0)[:,0]
    
    Selected_Users = torch.tensor(list(set(Candidate_Users_in_Train.numpy()).intersection(Candidate_Users_in_Test.numpy())))

    random_user_indices = torch.randperm(Selected_Users.numel(), generator=generator.manual_seed(SEED+FOLD))
    Test_Users = Selected_Users[random_user_indices[:min([random_user_indices.numel(), TEST_USER_COUNT])]]
    print("Test Users count: {}".format(len(Test_Users)))

    if SCENARIO=="cs":
        print("# convert test users to cold-start users in train set")
        print("# keep (>=5,10<=) items for test set users in train set")
        for tu in Test_Users:
            user_items = torch.nonzero(Train_Set[tu,:])[:,0]
            it_ids = torch.randperm(user_items.numel(), generator=generator.manual_seed(SEED+FOLD+int(tu)))
            # keepItemCount = round(user_items.numel() * 0.05)
            keepItemCount = torch.randint(5, 10, (1,), generator=generator.manual_seed(SEED+FOLD+int(tu)))
            removed_train_items = user_items[it_ids[:keepItemCount]]
            remove = torch.zeros(nM)
            remove[removed_train_items] = 1
            Train_Set[tu,:] = Train_Set[tu,:] * remove

    print("# keep max (5) test item for test set users in test set")
    # obtain test items max(1) for users  in test set
    max_test_item = 5
    for tu in Test_Users:
        candidate_items = torch.where(ProbeSet[tu, :] > 0)[0]
        item_ids = torch.randperm(candidate_items.numel(), generator=generator.manual_seed(SEED+FOLD+int(tu)))
        test_items = candidate_items[item_ids[:max_test_item]]
        keep = torch.zeros(nM)
        keep[test_items] = 1
        ProbeSet[tu, :] = ProbeSet[tu, :] * keep

    print("Test Users count: {}".format(len(Test_Users)))

    # keep test users, reset others
    Test_Set = torch.zeros(nU, nM)
    Test_Users = Test_Users.long()
    Test_Set[Test_Users, :] = 1
    Test_Set = Test_Set * ProbeSet    

    print("remove empty items")
    # remove empty items
    empty_items = torch.where(torch.sum(Train_Set, dim=0) == 0)[0]
    Train_Set = torch.index_select(Train_Set, dim=1, index=torch.tensor([i for i in range(nM) if i not in empty_items]))
    Test_Set = torch.index_select(Test_Set, dim=1, index=torch.tensor([i for i in range(nM) if i not in empty_items]))
    #nU, nM = Train_Set.shape

    Label_Set = np.delete(Label_Set, np.array(empty_items))
    print("remove empty test users and items recursively")
    
    # remove empty test users and items recursively
    stop_clear = False
    max_iterations = 100
    counter = 0
    while not stop_clear and counter < max_iterations:
        empty_test_users_in_test = torch.where(torch.sum(Test_Set[Test_Users, :], dim=1) <= 0)[0]
        empty_test_users_in_train = torch.where(torch.sum(Train_Set[Test_Users, :], dim=1) <= 0)[0]
        Test_Users = torch.tensor([tu for tu in Test_Users if tu not in empty_test_users_in_test and tu not in empty_test_users_in_train]) 
        if len(empty_test_users_in_test)>0:
            Test_Set[empty_test_users_in_test, :] = 0
        if len(empty_test_users_in_train)>0:
            Train_Set[empty_test_users_in_train, :] = 0
        
        empty_items = torch.where(torch.sum(Train_Set, dim=0) == 0)[0]
        if empty_items.numel()>0:
            Label_Set = np.delete(Label_Set, np.array(empty_items))
            Train_Set = torch.index_select(Train_Set, dim=1, index=torch.tensor([i for i in range(nM) if i not in empty_items]))
            Test_Set = torch.index_select(Test_Set, dim=1, index=torch.tensor([i for i in range(nM) if i not in empty_items]))                        
        
        if len(empty_test_users_in_test) + len(empty_test_users_in_train + empty_items.numel()) == 0:
            stop_clear = True
        
        counter += 1

    print("train data")
    # export train data
    train_data_filename = SCENARIO + '_' + DS_NAME + '_' + str(FOLD) + '.tr'
    train_file_path = DS_FOLDER + train_data_filename 
    rows1, cols1 = torch.nonzero(Train_Set, as_tuple=True)
    vals1 = Train_Set[rows1, cols1]
    train_data = torch.cat((rows1.unsqueeze(1), cols1.unsqueeze(1), vals1.unsqueeze(1)), dim=1)
    np.savetxt(train_file_path, train_data.numpy(), fmt='%d', delimiter='\t')
    print("train data written")

    print("test data")
    # export test data 
    test_data_filename = SCENARIO + '_' + DS_NAME + '_' + str(FOLD) + '.ts'
    test_file_path = DS_FOLDER + test_data_filename 
    rows2, cols2 = torch.nonzero(Test_Set, as_tuple=True)
    vals2 = Test_Set[rows2, cols2]
    test_data = torch.cat((rows2.unsqueeze(1), cols2.unsqueeze(1), vals2.unsqueeze(1)), dim=1)
    np.savetxt(test_file_path, test_data.numpy(), fmt='%d', delimiter='\t')
    print("test data written")

    print("label data")
    # export label data 
    label_data_filename = SCENARIO + '_' + DS_NAME + '_' + str(FOLD) + '.la'
    test_file_path = DS_FOLDER + label_data_filename 
    with open(test_file_path, "w") as file:
        counter=0
        for item in Label_Set:
            row_index = np.where(labels[:, 0] == str(int(item)))[0][0]
            movie_name = labels[row_index, 1]
            genre = labels[row_index, 2]
            file.write("%s::%s::%s::%s\n" % (counter, str(int(item)), movie_name, genre))    
            # file.write("%s,%s\n" % (item, item.upper()))
            counter = counter + 1
    print("label data written")