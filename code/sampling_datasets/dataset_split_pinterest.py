import torch
import random
import sys, os, builtins
import numpy as np
import pandas as pd

current_directory = os.path.dirname(__file__)

# path for raw datasets and outputs
DS_PATH = os.path.join(current_directory, "../raw_datasets/")
DS_FOLDER = os.path.join(current_directory, "../datasets/")

# Sampling parameters
# for users at least {mini} items clicked and maximum {maxi} items clicked
minu = 20
maxu = 20000
# for items at least clicked by {minu} users and clicked by maximum {maxu} users
mini = 5
maxi = 200

# we removed ratings under 1 from dataset
min_rating = 1

if os.getenv('USER')=="hyilmazer":
    sys.stdout = open("/truba_scratch/hyilmazer/benchmarks/ipana/pinterest.txt".format(minu,maxu,mini,maxi,min_rating), "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)

# output dataset file prefix
DS_NAME = 'pinterest'

# scenario
# cs : cold-start
# lt : long-tail
#  cs | lt
SCENARIO = "cs" 

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

## ## ## 
# # Extract csv file to panda
tp = pd.read_csv(os.path.join(DS_PATH, "pinterest-20.dat"), header=None, sep='\t', usecols=[0, 1, 2], names=['uid', 'sid', 'rating'])
# unique uids
unique_uid_count = tp['uid'].nunique()

# unique sids
unique_sid_count = tp['sid'].nunique()

# total rating
total_rows = tp.shape[0]

print(f"Unique uid count: {unique_uid_count}")
print(f"Unique sid count: {unique_sid_count}")
print(f"Total rating count: {total_rows}")
orig_ratio = total_rows/(unique_uid_count*unique_sid_count)
print(f"Original Sparsity: { str((1-orig_ratio)*100)}")

def filter_triplets(df, minu, maxu, mini, maxi, min_rating):
    df_old = []
    while True:
        df = df[df['rating'] >= min_rating]

        item_frequencies = df['sid'].value_counts()
        df = df[df['sid'].isin(item_frequencies[item_frequencies.between(mini, maxi)].index)]

        user_frequencies = df['uid'].value_counts()
        df = df[df['uid'].isin(user_frequencies[user_frequencies.between(minu, maxu)].index)]

        # Eğer veri setinde değişiklik olmazsa döngüyü sonlandır
        if len(df) == len(df_old):
            break

        df_old = df.copy()
    df_old = []
    return df

# first filter before recursively
tp = filter_triplets(tp, minu,maxu,mini,maxi,min_rating)

###### Numerize
# Uniqu ids
unique_uid = tp['uid'].unique()
unique_sid = tp['sid'].unique()

# Numeric ids
user2id = {uid: i for i, uid in enumerate(unique_uid)}
song2id = {sid: i for i, sid in enumerate(unique_sid)}

# Filter ids
tp['uid'] = tp['uid'].map(user2id)
tp['sid'] = tp['sid'].map(song2id)

# write sampled csv file as new war file
data = tp.values
tp = None

# Mapping 
unique_user_ids = np.unique(data[:, 0])
unique_movie_ids = np.unique(data[:, 1])
user_map = {id: i for i, id in enumerate(unique_user_ids)}
movie_map = {id: i for i, id in enumerate(unique_movie_ids)}

# Indices
user_indices = torch.LongTensor([user_map[id] for id in data[:, 0]])
movie_indices = torch.LongTensor([movie_map[id] for id in data[:, 1]])
indices = torch.stack([user_indices, movie_indices], dim=0)

# some minor fixes 
count_of_count_values = len(data[:, 2][data[:, 2] == 'count'])
values = torch.FloatTensor(data[:, 2].astype(np.float32))

# Convert listen counts under threshold to zero others 1
size = (len(unique_user_ids), len(unique_movie_ids))
RR = torch.sparse.FloatTensor(indices, values, size)
RR_dense = RR.to_dense()
RR_dense[RR_dense < min_rating] = 0
RR_dense[RR_dense >= min_rating] = 1
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

print("{} {} {} {} {}".format(minu,maxu,mini,maxi,min_rating))

# iterate until the condition is met
while not stop_clear and counter < max_iterations:
    # 
    row_sums = torch.sum(RR, dim=1)
    col_sums = torch.sum(RR, dim=0)

    # 
    row_mask = (row_sums >= minu) & (row_sums <= maxu)
    col_mask = (col_sums >= mini) & (col_sums <= maxi)

    rsum = int(row_mask.sum())
    csum = int(col_mask.sum())

    if rsum < nU:
        RR = RR[row_mask,:]

    if csum < nM:
        RR = RR[:,col_mask]

    if (rsum+csum) == (nU+nM):
        stop_clear = True      

    nU, nM = RR.shape
    nU = int(nU)
    nM = int(nM)

    counter += 1

    print(RR.shape)

# 
ones_count = np.count_nonzero(RR == 1)
# 
zeros_count = np.count_nonzero(RR == 0)
# Sparsity
ratio = ones_count / (nU*nM)

print("Clicks:"+str(ones_count))
print("All:"+ str(zeros_count))
print("Sparsity:"+ str((1-ratio)*100))


# Create Folds
##########################################
# create raw train set
Raw_Set = RR.clone()
nU, nM = Raw_Set.shape
# create probe set for selected fold
ProbeSetRand = torch.rand(nU, nM)
RawProbeSet = Raw_Set * ProbeSetRand

# Find Item Frequencies
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

    generator = torch.Generator()

    # CV range in random 
    cval = 1 / CV_FOLD
    minr = (FOLD-1) * cval
    maxr = cval + minr

    ProbeSet = RawProbeSet.clone()
    mask = torch.logical_or(ProbeSet < minr, ProbeSet >= maxr)
    ProbeSet[mask] = 0    
    ProbeSet[ProbeSet>0] = 1
    ProbeSet = ProbeSet.int()

    # remove probe set from raw set
    Train_Set = Raw_Set - ProbeSet

    nU, nM = Train_Set.shape

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
        # keep (>=5,10<=) items for test set users in train set
        for tu in Test_Users:
            user_items = torch.nonzero(Train_Set[tu,:])[:,0]
            it_ids = torch.randperm(user_items.numel(), generator=generator.manual_seed(SEED+FOLD+int(tu)))
            # keepItemCount = round(user_items.numel() * 0.05)
            keepItemCount = torch.randint(5, 10, (1,), generator=generator.manual_seed(SEED+FOLD+int(tu)))
            removed_train_items = user_items[it_ids[:keepItemCount]]
            remove = torch.zeros(nM)
            remove[removed_train_items] = 1
            Train_Set[tu,:] = Train_Set[tu,:] * remove

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

    # remove empty items
    empty_items = torch.where(torch.sum(Train_Set, dim=0) == 0)[0]
    Train_Set = torch.index_select(Train_Set, dim=1, index=torch.tensor([i for i in range(nM) if i not in empty_items]))
    Test_Set = torch.index_select(Test_Set, dim=1, index=torch.tensor([i for i in range(nM) if i not in empty_items]))

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
        if len(empty_test_users_in_test) + len(empty_test_users_in_train) == 0:
            stop_clear = True
        counter += 1

    # write train data
    train_data_filename = SCENARIO + '_' + DS_NAME.split('.')[0] + '_' + str(FOLD) + '.tr'
    train_file_path = DS_FOLDER + train_data_filename 
    rows1, cols1 = torch.nonzero(Train_Set, as_tuple=True)
    vals1 = Train_Set[rows1, cols1]
    train_data = torch.cat((rows1.unsqueeze(1), cols1.unsqueeze(1), vals1.unsqueeze(1)), dim=1)
    np.savetxt(train_file_path, train_data.numpy(), fmt='%d', delimiter='\t')
    print("train set written")

    # write test data 
    test_data_filename = SCENARIO + '_' + DS_NAME.split('.')[0] + '_' + str(FOLD) + '.ts'
    test_file_path = DS_FOLDER + test_data_filename 
    rows2, cols2 = torch.nonzero(Test_Set, as_tuple=True)
    vals2 = Test_Set[rows2, cols2]
    test_data = torch.cat((rows2.unsqueeze(1), cols2.unsqueeze(1), vals2.unsqueeze(1)), dim=1)
    np.savetxt(test_file_path, test_data.numpy(), fmt='%d', delimiter='\t')
    print("test set written")