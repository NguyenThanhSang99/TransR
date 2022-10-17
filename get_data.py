
file_list = ["entities", "relations", "triple"]

entities = {}
relations = {}

for file in file_list:
    file1 = open('data/{}.txt'.format(file), 'r', encoding='utf-8')
    file2 = open('data/new_{}.txt'.format(file), 'w', encoding='utf-8')

    lines = file1.readlines()

    file2.writelines(str(len(lines)) + "\n")

    for line in lines:
        line_list = line.strip().split("\t")
        if (file == "entities"):
            entities[line_list[1]] = line_list[0]
        elif (file == "relations"):
            relations[line_list[1]] = line_list[0]
        else:
            file2.writelines(entities[line_list[0]] + " " +
                             entities[line_list[2]] + " " + relations[line_list[1]] + "\n")
            continue
        file2.writelines(line_list[1] + "\t" + line_list[0] + "\n")

    file1.close()
    file2.close()

print(entities)
