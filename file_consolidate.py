import os

current_path = os.getcwd() #relative to file placement, make sure all data files from original download in folder in same directory

def consolidate_files(input_path, out_file_name): #concatenates data from files in input_path and puts it in file called out_file_name
	local_path = current_path + '\\' + input_path #path to input files
	with open(current_path + '\\' + out_file_name, mode='w', encoding='utf-8') as out_file:
		idx = 0 #counter for number of files
		for idx, filename in enumerate(os.listdir(local_path)): #iterate 
			with open(os.path.join(local_path, filename), 'r', encoding='utf-8') as f:
				text = f.read()
				score = filename.split('_')[1][:-4] # extracts score from file name
				out_file.write(score + '\t' + text + '\n')
	print(f'Finished consolidating {idx+1} files from {input_path} to {out_file_name}')

def consolidate_files_split(input_path, out_file_1_name, out_file_2_name): #concatenates data from files in input_path and splits it into 2 files: out_file_1_name, out_file_2_name
	local_path = current_path + '\\' + input_path
	with open(current_path + '\\' + out_file_1_name, mode='w', encoding='utf-8') as out_file_1, open(current_path + '\\' + out_file_2_name, mode='w', encoding='utf-8') as out_file_2:
			idx = 0 #counter for number of files
			for idx, filename in enumerate(os.listdir(local_path)): #iterate through files
				with open(os.path.join(local_path, filename), 'r', encoding='utf-8') as f:
					text = f.read()
					score = filename.split('_')[1][:-4] # extracts score from file name
					if(idx % 2 == 0):
						out_file_1.write(score + '\t' + text + '\n')
					else:
						out_file_2.write(score + '\t' + text + '\n')
	print(f'Finished consolidating {idx+1} files from {input_path} to {out_file_1_name} and {out_file_2_name}')

if __name__ == '__main__':
	consolidate_files("aclImdb_v1\\aclImdb\\train\\pos", 'train_pos_reviews.txt')
	consolidate_files("aclImdb_v1\\aclImdb\\train\\neg", 'train_neg_reviews.txt')
	consolidate_files_split("aclImdb_v1\\aclImdb\\test\\pos", 'dev_pos_reviews.txt', 'test_pos_reviews.txt')
	consolidate_files_split("aclImdb_v1\\aclImdb\\test\\neg", 'dev_neg_reviews.txt', 'test_neg_reviews.txt')


