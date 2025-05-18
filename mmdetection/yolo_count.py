import os

def count_yolo_labels(label_dir):
    class_counts = {}
    total_labels = 0
    num_files = 0

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        file_path = os.path.join(label_dir, label_file)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                num_labels_in_file = len(lines)
                total_labels += num_labels_in_file

                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                        else:
                            class_counts[class_id] = 1

                num_files += 1
        except Exception as e:
            print(f"Error reading {label_file}: {e}")

    sorted_class_counts = dict(sorted(class_counts.items()))

    return {
        "total_files": num_files,
        "total_labels": total_labels,
        "class_counts": sorted_class_counts
    }

def count_dataset_labels(dataset_path):
    results = {}
    
    # 检查是否有训练集、验证集和测试集
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if os.path.exists(subset_path):
            labels_path = os.path.join(subset_path, 'labels')
            if os.path.exists(labels_path):
                subset_results = count_yolo_labels(labels_path)
                results[subset] = subset_results
    
    # 添加总统计
    total_files = sum(results[subset]["total_files"] for subset in results)
    total_labels = sum(results[subset]["total_labels"] for subset in results)
    
    # 合并所有子集的类别统计
    total_class_counts = {}
    for subset in results:
        for class_id, count in results[subset]["class_counts"].items():
            if class_id in total_class_counts:
                total_class_counts[class_id] += count
            else:
                total_class_counts[class_id] = count
    
    results["total"] = {
        "total_files": total_files,
        "total_labels": total_labels,
        "class_counts": dict(sorted(total_class_counts.items()))
    }
    
    return results

def print_dataset_stats(stats):
    print("\n" + "="*50)
    print(f"{'Dataset Statistics':^50}")
    print("="*50 + "\n")
    
    for subset in ['train', 'val', 'test']:
        if subset in stats:
            print(f"{subset.capitalize()} set statistics:")
            print(f"  Number of files: {stats[subset]['total_files']}")
            print(f"  Number of labels: {stats[subset]['total_labels']}")
            print("  Class distribution:")
            for class_id, count in stats[subset]['class_counts'].items():
                print(f"    Class {class_id}: {count} labels")
            print("-"*50)
    
    print(f"{'Total statistics':^50}")
    print(f"  Total number of files: {stats['total']['total_files']}")
    print(f"  Total number of labels: {stats['total']['total_labels']}")
    print("  Total class distribution:")
    for class_id, count in stats['total']['class_counts'].items():
        print(f"    Class {class_id}: {count} labels")
    print("="*50 + "\n")

if __name__ == "__main__":
    # 替换为你的YOLO数据集根目录
    dataset_root = "/data/hyt/yolo/datasets/Darbhanga"
    
    dataset_stats = count_dataset_labels(dataset_root)
    print_dataset_stats(dataset_stats)