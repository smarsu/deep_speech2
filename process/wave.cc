// Copyright 2019 smarsu. All Rights Reserved.

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstdlib>

extern "C" {

struct wave {
    char ChunkID[4];  // b "RIFF"
    uint32_t ChunkSize;  // l 4 + (8 + SubChunk1Size) + (8 + SubChunk2Size)
    char Format[4];  // b "WAVE"

    char Subchunk1ID[4];  // b "fmt "
    uint32_t Subchunk1Size;  // l 16 for PCM
    uint16_t AudioFormat;  // l 1 not compression
    uint16_t NumChannels;  // l 1, 2
    uint32_t SampleRate;  // l 16000, ...
    uint32_t ByteRate;  // l SampleRate * NumChannels * BitsPerSample / 8
    uint16_t BlockAlign;  // l NumChannels * BitsPerSample / 8
    uint16_t BitsPerSample;  // l 8 bits = 8, 16 bits = 16

    char Subchunk2ID[4];  // b "data"
    uint32_t Subchunk2Size;  // l NumSamples * NumChannels * BitsPerSample / 8

    // void *data;
}; 

typedef struct wave wave_t;

void ReadWave(const char *wave_path, 
              int *nchannels, 
              int *sampwidth,
              int *framerate,
              int *nframes,
              void **data);

void CptrFree(void *cptr);

}  // extern "C"

struct Buffer {
    Buffer() {}

    void resize(size_t size) {
        if (size > capacity_) {
            if (data_) {
                free(data_);
            }
            data_ = malloc(size);
            assert(data_);
            capacity_ = size;
        }
    }

    ~Buffer() {
        if (data_) {
            free(data_);
        }
    }

    void *data_{nullptr};
    size_t capacity_{0};
};

Buffer buffer;


bool str_equal(const char *x, 
               const char *y, 
               size_t size) {
    for (int i = 0; i < size; ++i) {
        if (x[i] != y[i]) {
            printf("x: %c, y: %c\n", x[i], y[i]);
            return false;
        }
    }
    return true;
}

int32_t swapInt32(int32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0xFF000000) >> 24) ;
}

int16_t swapInt16(int16_t value) {
    return ((value & 0x00FF) << 8) |
           ((value & 0xFF00) >> 8);
}

void CptrFree(void *cptr) {
    printf("release %p\n", cptr);
    free(cptr);
}

void ReadWave(const char *wave_path, 
              int *nchannels, 
              int *sampwidth,
              int *framerate,
              int *nframes,
              void **data) {
    FILE *fb = fopen(wave_path, "rb");
    assert(fb);

    wave_t wav;
    char *wav_ = reinterpret_cast<char *>(&wav);

    fread(wav_, 1, 20, fb);
    assert(str_equal(wav.ChunkID, "RIFF", 4));
    assert(str_equal(wav.Format, "WAVE", 4));
    assert(str_equal(wav.Subchunk1ID, "fmt ", 4));
    assert(wav.Subchunk1Size == 16);

    fread(wav_ + 20, 1, 16, fb);
    assert(wav.AudioFormat == 1);

    fread(wav_ + 36, 1, 8, fb);
    assert(str_equal(wav.Subchunk2ID, "data", 4));

    size_t NumSamples = wav.Subchunk2Size * 8 / wav.BitsPerSample / wav.NumChannels;

    buffer.resize(wav.Subchunk2Size);

    // void *wavsignal = malloc(wav.Subchunk2Size);
    // assert(wavsignal);
    fread(reinterpret_cast<char *>(buffer.data_), 
          1,
          wav.Subchunk2Size, 
          fb);

    *nchannels = wav.NumChannels;
    *sampwidth = wav.BitsPerSample;
    *framerate = wav.SampleRate;
    *nframes = NumSamples;
    *data = buffer.data_;

    fclose(fb);
}
