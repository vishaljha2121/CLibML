#include "../../include/tensorNew.h"
#include "../../include/err.h"
#include <stdio.h>
#include <stdlib.h>

// Helper to swap endianness if needed (NMIST is big-endian, intel/arm usually little)
static u32 _swap_endian(u32 val) {
    return ((val << 24) & 0xFF000000) |
           ((val <<  8) & 0x00FF0000) |
           ((val >>  8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

// Basic .mat loader for compatibility with existing data
tensor* tensor_load_mat(mg_arena* arena, string8 file_path, u32 rows, u32 cols) {
    FILE* f = fopen((char*)file_path.str, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Limit size
    u64 expected = (u64)rows * cols * sizeof(f32);
    size = size < expected ? size : expected;

    tensor_shape shape = { cols, rows, 1 }; // Tensor is usually (Width, Height, Depth)
    // Matrix rows=Height, cols=Width.
    // If we map rows->height, cols->width
    shape = (tensor_shape){ cols, rows, 1 };
    
    // HOWEVER, for MNIST data in this project:
    // train_images.mat: 60000 rows, 784 cols.
    // Each row is an image (flattened).
    // So shape should be: Width=784, Height=60000 (Batch)? 
    // network.h says: "One training input is a 2D slice... Depth must be same as depth of train_inputs" - this is still confusing.
    // Let's assume the network expects (784, 1, 60000) or similar if using dense layers first.
    // The user's layout has `input shape = (28, 28, 1)`.
    // So we need to reshape the loaded data.
    // `mat_load` loads into a flat array.
    // We should load it into a tensor of shape (28, 28, 60000).
    // 60000 images. Each image 28x28.
    
    // Special case for MNIST MAT files
    if (cols == 784) {
         shape = (tensor_shape){ 28, 28, rows };
    } else if (cols == 1) {
        // Labels
        shape = (tensor_shape){ 10, 1, rows }; // 10 classes
    } else {
        // Fallback
        shape = (tensor_shape){ cols, rows, 1 };
    }
    
    tensor* out = tensor_create(arena, shape);
    if (!out) { fclose(f); return NULL; }
    
    if (cols == 1) { // Labels need expansion
        // Read floats (labels are 0.0-9.0 floats in the mat file?)
        // wait, `main.c` legacy:
        /*
        matrix* train_labels_file = mat_load(perm_arena, 60000, 1, "data/mnist/train_labels.mat");
        matrix* train_labels = mat_create(perm_arena, 60000, 10);
        for (u32 i = 0; i < 60000; i++) {
            u32 num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f;
        }
        */
        // So the .mat file contains FLOATS representing the index (0.00, 1.00 etc).
        
        // We need to read into temp buffer first
        f32* temp = malloc(size); // size bytes
        fread(temp, 1, size, f);
        fclose(f);
        
        tensor_fill(out, 0.0f);
        f32* out_data = (f32*)out->data;
        u32 count = size / sizeof(f32);
        for (u32 i = 0; i < count; i++) {
            u32 label = (u32)temp[i];
            if (label < 10) {
                 out_data[label + i * 10] = 1.0f;
            }
        }
        free(temp);
    } else {
        // Images: Read directly
        // But we might need to transpose if order differs?
        // Row-major (C) vs Column-major (Fortran)?
        // `mat_load` just reads bytes.
        // If data is (60000, 784), likely row-major.
        // Tensor is column-major? 
        // `tensor` struct usually implies some ordering.
        // Let's assume trivial copy for now.
        READ_DATA:
        fread(out->data, 1, size, f);
        fclose(f);
    }
    
    return out;
}

tensor* tensor_load_mnist_images(mg_arena* arena, string8 file_path, u32 num_images) {
    // Try MAT load first if extension is .mat
    if (file_path.size > 4 && 
        file_path.str[file_path.size-4] == '.' &&
        file_path.str[file_path.size-3] == 'm' &&
        file_path.str[file_path.size-2] == 'a' &&
        file_path.str[file_path.size-1] == 't') {
        return tensor_load_mat(arena, file_path, num_images, 784);
    }

    FILE* f = fopen((char*)file_path.str, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %.*s\n", (int)file_path.size, file_path.str);
        return NULL;
    }
    
    // Read header
    // magic (4), num_images (4), rows (4), cols (4)
    u32 header[4];
    if (fread(header, sizeof(u32), 4, f) != 4) {
         fclose(f);
         return NULL;
    }
    
    // Verify magic? 2051 for images
    // u32 magic = _swap_endian(header[0]);
    // u32 cols = _swap_endian(header[3]);
    
    // We assume standard MNIST 28x28
    fseek(f, 16, SEEK_SET); // Skip header
    
    // Shape: (28, 28, 1) per image.
    // We need to load `num_images`. 
    // The `tensor` structure supports 3D.
    // IF we are loading a BATCH of images for training:
    // `network_train` expects `train_inputs` to be a tensor where we take 2D slices?
    // See `network.h`: "One training input is a 2D slice of the `train_inputs` tensor. Use 2D slices... Depth must be same as depth of train_inputs"
    // Wait. `train_inputs` is a tensor. "One training input is a 2D slice" -> implies depth is the batch/count?
    // "If you have 3D inputs [e.g. RGB], reduce them to 2D for `train_inputs` then resize them in the input layer"
    // A bit confusing.
    // "2D slices are taken for each output."
    // Let's look at `network_train` impl in `network.c` if possible, OR assume standard convention:
    // Usually Batch is stored in Depth if Width/Height are spatial?
    // OR if single tensor stores ALL data: 
    // `network.h` says: "One training input is a 2D slice... Depth must be same as depth of train_inputs"
    // This implies `train_inputs` has shape (W, H, N). Slicing at Z=i gives the i-th sample (W, H).
    // So for 60000 images of 28x28, shape is (28, 28, 60000).
    
    tensor_shape shape = { 28, 28, num_images };
    // But MNIST file is u8 pixels. Tensor is f32.
    
    tensor* out = tensor_create(arena, shape);
    
    // Read directly into a temp buffer then convert?
    u64 size = (u64)28 * 28 * num_images;
    u8* buf = malloc(size); // Temp malloc
    if (fread(buf, 1, size, f) != size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    
    // Convert to f32 0.0-1.0
    f32* data = (f32*)out->data;
    for (u64 i = 0; i < size; i++) {
        data[i] = (f32)buf[i] / 255.0f;
    }
    
    free(buf);
    return out;
}

tensor* tensor_load_mnist_labels(mg_arena* arena, string8 file_path, u32 num_labels) {
    if (file_path.size > 4 && 
        file_path.str[file_path.size-4] == '.' &&
        file_path.str[file_path.size-3] == 'm' &&
        file_path.str[file_path.size-2] == 'a' &&
        file_path.str[file_path.size-1] == 't') {
        return tensor_load_mat(arena, file_path, num_labels, 1);
    }

    FILE* f = fopen((char*)file_path.str, "rb");
    if (!f) return NULL;
    
    fseek(f, 8, SEEK_SET); // Skip header (magic, num_items)
    
    u8* buf = malloc(num_labels);
    if (fread(buf, 1, num_labels, f) != num_labels) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    
    // Labels need to be one-hot encoded? 
    // `network.h`: "Training outputs... 2D slices... Depth must be same as depth of train_inputs"
    // Output layer for MNIST is usually 10.
    // So shape should be (10, 1, num_labels)? 
    // Or (1, 10, num_labels)?
    // Dense layer output is usually (Size, 1, 1) or (1, Size, 1)?
    // Dense layer `feedforward`: `_layer_dense_feedforward`
    // It creates output of shape `(l->dense.size, 1, 1)`?
    // Checking `layers_dense.c` would confirm.
    // Assuming (10, 1) per sample.
    // So total tensor shape: (10, 1, num_labels).
    
    tensor_shape shape = { 10, 1, num_labels };
    tensor* out = tensor_create(arena, shape);
    tensor_fill(out, 0.0f);
    f32* data = (f32*)out->data;
    
    for (u32 i = 0; i < num_labels; i++) {
        u8 label = buf[i];
        if (label < 10) {
            // Index in flat array: 
            // z=i, y=0. x varies 0..9
            // index = x + y*w + z*w*h
            //       = label + 0 + i*10*1
            data[label + i * 10] = 1.0f;
        }
    }
    
    free(buf);
    return out;
}
