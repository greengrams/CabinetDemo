import numpy as np

def envelope(train_label, train_data, test_data, axis_en_params):
    unique_label = set(train_label)
    s_train = [len(train_data), np.max([np.max(map(lambda x: len(x), train_data)),len(test_data[0])])]
    s_test  = [len(test_data),len(test_data[0])]
    s_label = len(unique_label)
    mean_train = np.zeros([s_label, s_train[1]])
    std_train = np.zeros([s_label, s_train[1]])

    # print unique_label
    for i in range(s_label):
        now_label_data = []
        label = unique_label.pop()
        # Find the same label's data
        for t in range(len(train_data)):
                if train_label[t] == label:
                        now_label_data.append(train_data[t])
        # Compute the mean
        mean_train[i][:len(now_label_data[0])] = np.mean(now_label_data, axis=0)

        # Compute the std
        std_train[i][:len(now_label_data[0])]  = np.std(now_label_data, axis=0)

    envelope_data = np.zeros([s_test[0], 3 * s_label])
        
    if len(axis_en_params)==0:
        all_entropy=[]
        std_list=[]
        for count in xrange(s_label):
            std_list=[]
            label_entropy=[]
            # k_std 0.1 to 3.0
            for i in xrange(30):
                k_std=float(i+1)/10
                std_list.append(k_std)
                
                now_label_data=np.array(train_data)[train_label==count]
                num_one = np.sum(now_label_data > mean_train[count][:s_train[1]] + k_std * std_train[count][:s_train[1]], 1,dtype=float)
                num_minus = np.sum(now_label_data < mean_train[count][:s_train[1]] - k_std * std_train[count][:s_train[1]], 1,dtype=float)
                num_zero = np.array(s_test[1] - num_one - num_minus,dtype=float)
                p=np.array([num_one,num_minus,num_zero])/s_test[1]
                ## avoid 0 to let entropy as nan
                p[p==0]=1.0
                std_entropy=np.mean(np.sum(np.array(-p*np.log(p)/np.log(2)),0))
                
                label_entropy.append(std_entropy)
            all_entropy.append(label_entropy)
        all_entropy=np.array(all_entropy)
        std_list=np.array(std_list)
        # print all_entropy
        
        lambda_=3
        best_std_list=[]
        ## get envelope feature by best k-std
        for count in xrange(s_label):
            # print 'count'+str(count)
            target_entropy=all_entropy[count]
            other_entropy=all_entropy[np.array(range(s_label))!=count]
            obj_val=target_entropy-lambda_*(np.min(other_entropy,0)-target_entropy)
            # print obj_val
            best_std=std_list[np.where(obj_val==min(obj_val))]
            # print best_std
            best_std_list.append(float(best_std))
            
            num_one   = np.sum(test_data > mean_train[count][:len(test_data[0])] + best_std * std_train[count][:len(test_data[0])], 1)
            num_minus = np.sum(test_data < mean_train[count][:len(test_data[0])] - best_std * std_train[count][:len(test_data[0])], 1)
            envelope_data[:, (count)*3] = s_test[1] - num_one - num_minus
            envelope_data[:, (count)*3 + 1] = num_one
            envelope_data[:, (count)*3 + 2] = num_minus
        print best_std_list
        axis_en_params.extend(best_std_list)
    else:
        for count in xrange(s_label):
            num_one   = np.sum(test_data > mean_train[count][:len(test_data[0])] + axis_en_params[count] * std_train[count][:len(test_data[0])], 1)
            num_minus = np.sum(test_data < mean_train[count][:len(test_data[0])] - axis_en_params[count] * std_train[count][:len(test_data[0])], 1)
            envelope_data[:, (count)*3] = s_test[1] - num_one - num_minus
            envelope_data[:, (count)*3 + 1] = num_one
            envelope_data[:, (count)*3 + 2] = num_minus
    return envelope_data

if __name__ == '__main__':
    train_label = [1,1,2,2]
    train_data = [[1,0,1], [1,1,1], [0,0,1,1], [1,0,1,1]]
    test_data = [[1,1,1], [1,2,2]]
    num_std = 1
    print envelope(train_label, train_data, test_data, num_std)


