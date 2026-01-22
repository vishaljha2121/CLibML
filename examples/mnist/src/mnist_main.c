//
// Created by Vishal Jha on 19/01/26.
//
#include <stdio.h>
#include <string.h>
#include <mlframework/network.h>
#include <mlframework/base_defs.h>
#include <mlframework/mg_arena.h>

void print_usage() {
    printf("Usage:\n");
    printf("  MLFramework train <layout.tsl> <data_dir> <train_desc.tsd>\n");
    printf("  MLFramework infer <model.tsn> <input_file>\n");
    printf("  MLFramework snake [train|play] <model_path?>\n");
}

// Forward declare snake main
int snake_main(int argc, char** argv);

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }
    
    mga_desc desc = { .desired_max_size = MGA_MiB(512), .desired_block_size = MGA_MiB(4) };
    mg_arena* arena = mga_create(&desc);
    
    char* command = argv[1];
    if (strcmp(command, "train") == 0) {
        if (argc < 5) {
            printf("Error: Missing arguments for train.\n");
            print_usage();
            return 1;
        }
        
        char* layout_file = argv[2];
        char* data_dir = argv[3];
        char* desc_file = argv[4];
        
        printf("Loading layout from %s...\n", layout_file);
        network* nn = network_load_layout(arena, str8_from_cstr((u8*)layout_file), true);
        if (!nn) {
            printf("Failed to load network layout.\n");
            return 1;
        }
        
        printf("Loading training description from %s...\n", desc_file);
        network_train_desc desc = {0};
        
        // Read file content
        FILE* f = fopen(desc_file, "rb");
        if (!f) { printf("Cannot open desc file\n"); return 1; }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        u8* buf = mga_push(arena, fsize + 1);
        fread(buf, 1, fsize, f);
        buf[fsize] = 0;
        fclose(f);
        
        train_desc_load(&desc, str8_from_cstr(buf));
        
        printf("Loading data from %s...\n", data_dir);
        // Construct paths
        string8 train_img_path = str8_pushf(arena, "%s/train_images.mat", data_dir);
        string8 train_lbl_path = str8_pushf(arena, "%s/train_labels.mat", data_dir);
        string8 test_img_path = str8_pushf(arena, "%s/test_images.mat", data_dir);
        string8 test_lbl_path = str8_pushf(arena, "%s/test_labels.mat", data_dir);
        
        // Load Data
        desc.train_inputs = tensor_load_mnist_images(arena, train_img_path, 60000);
        desc.train_outputs = tensor_load_mnist_labels(arena, train_lbl_path, 60000);
        
        // Optional Test Data
        desc.test_inputs = tensor_load_mnist_images(arena, test_img_path, 10000);
        desc.test_outputs = tensor_load_mnist_labels(arena, test_lbl_path, 10000);
        
        if (desc.test_inputs && desc.test_outputs) {
            desc.accuracy_test = true;
        }
        
        if (!desc.train_inputs || !desc.train_outputs) {
             printf("Failed to load training data.\n");
             return 1;
        }
        
        printf("Starting training...\n");
        network_train(nn, &desc);
        
        printf("Training complete.\n");
        if (desc.save_path.size > 0) {
            // network_train might save checkpoints, but let's save final model
            string8 final_path = str8_pushf(arena, "%.*s_final.tsn", (int)desc.save_path.size, desc.save_path.str);
            network_save(nn, final_path);
            printf("Saved model to %.*s\n", (int)final_path.size, final_path.str);
        } else {
             network_save(nn, STR8("model_final.tsn"));
             printf("Saved model to model_final.tsn\n");
        }

    } else if (strcmp(command, "infer") == 0) {
        if (argc < 4) {
             printf("Error: Missing arguments for infer.\n");
             print_usage();
             return 1;
        }
        char* model_file = argv[2];
        char* input_file = argv[3]; // Assume serialized tensor? or raw image?
        // TODO: Implement Inference loading
        
        printf("Loading model from %s...\n", model_file);
        network* nn = network_load(arena, str8_from_cstr((u8*)model_file), false);
        if (!nn) {
             printf("Failed to load model.\n");
             return 1;
        }
        
        // Load input (assuming single MNIST image for Demo)
        // Or specific input format?
         // For simplicity, let's just make a dummy input or load one from file
         printf("Inference on %s not fully implemented for generic files yet.\n", input_file);
    } else if (strcmp(command, "snake") == 0) {
        return snake_main(argc, argv);
    } else {
        print_usage();
    }
    
    mga_destroy(arena);
    return 0;
}