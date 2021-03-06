def get_training_data(train_data_path):
    """
        input: 
            ---------------
            抱 B-v O
            怨 B-v O    -> sentence 1
            的 b-u O

            科 B-n O
            技 B-n O    -> sentence 2
            ---------------
    """
    data = open(train_data_path).readlines()
    training_data = []

    list1 = []
    list2 = []
    for i in range(len(data)):
        if data[i] != '\n':
            char, pos, err = data[i].split()
            list1.append(char)
            list2.append(err)
        else:
            training_data.append((list1, list2))
            list1 = []
            list2 = []
    return training_data

def color_tag(train_data):
    all_tag_nums = []
    for i in range(len(train_data)):
        seq = train_data[i][0]
        tag = train_data[i][1]
        tag_nums = [0, 0, 0, 0] #R,M,S,W
        
        for j in range(len(tag)):
            err_type = tag[j].split('-')[-1]
            if err_type == 'R':
                seq[j] = "\033[1;31;40m"+seq[j]+"\033[0m"   #red
                tag_nums[0] += 1
            if err_type == 'M':
                seq[j] = "\033[1;30;44m"+seq[j]+"\033[0m"   #blue
                tag_nums[1] += 1
            if err_type == 'S':
                seq[j] = "\033[1;30;43m"+seq[j]+"\033[0m"   #yellow
                tag_nums[2] += 1
            if err_type == 'W':
                seq[j] = "\033[1;30;45m"+seq[j]+"\033[0m"   #magenta
                tag_nums[3] += 1
        all_tag_nums.append(tag_nums)
        print(i,''.join(seq))
    print('Done!')

if __name__ == '__main__':
    train_data_path = 'train_CGED2016.txt'
    train_data = get_training_data(train_data_path)
    test_data_path = 'test_CGED2016.txt' 
    test_data = get_training_data(test_data_path)
    color_tag(train_data)
