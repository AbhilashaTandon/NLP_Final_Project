import os

current_path = os.getcwd() #relative to file placement, make sure all data files from original download in folder in same directory

def consolidate_file(path, new_file): #takes path to input files and name of output file as arguments, returns no arguments but creates new file in current directory
	all_files = "" #string containing data to write to output
	i = 0 #counter for number of files
	for filename in os.listdir(current_path + path): #iterate through files
		with open(os.path.join(current_path + path, filename), 'r', encoding='utf-8') as f:
			text = f.read()
			score = filename.split('_')[1][:-4] # extracts score from file name
			all_files += (score + '\t' + text + '\n')
			i += 1
	print(f'Finished consolidating {i} files from {path} to {new_file}')
	with open(current_path + new_file, mode='w', encoding='utf-8') as output_file:
		output_file.write(all_files)

if __name__ == '__main__':
	consolidate_file("\\aclImdb_v1\\aclImdb\\train\\pos", 'train_pos_reviews.txt')
	consolidate_file("\\aclImdb_v1\\aclImdb\\train\\neg", 'train_neg_reviews.txt')
	consolidate_file("\\aclImdb_v1\\aclImdb\\test\\pos", 'test_pos_reviews.txt')
	consolidate_file("\\aclImdb_v1\\aclImdb\\test\\neg", 'test_neg_reviews.txt')