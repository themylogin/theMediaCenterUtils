#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <pthread.h>

#include <Imlib2.h>
#include <X11/Xlib.h>

#include "image.cuh"
#include "lastfm.h"
#include "mpd.h"

Display* display;
Window root;
Screen* screen;
Pixmap pixmap;

DATA32* dev_im1data;
DATA32* dev_im2data;
DATA32* dev_im3data;

Imlib_Image default_image;

DATA32* transitionImageData[25];
Imlib_Image transitionImage[25];

void set_image(char* image)
{
    Imlib_Image new_image;
    // После этого блока должен быть инициализирован new_image размера screen->width x screen->height и в dev_im1data лежать его данные
    if (image != NULL)
    {
        Imlib_Image new_image_unprepared = imlib_load_image_immediately_without_cache(image);
        imlib_context_set_image(new_image_unprepared);
        int src_width = imlib_image_get_width();
        int src_height = imlib_image_get_height();

        int new_width, new_height;
        // Сначала растягиваем во весь экран
        if ((float)src_width / (float)src_height > (float)screen->width / (float)screen->height)
        {
            new_width = screen->width;
            new_height = (float)new_width / (float)src_width * (float)src_height;
        }
        else
        {
            new_height = screen->height;
            new_width = (float)new_height / (float)src_height * (float)src_width;
        }
        // Но если растянули больше, чем в 2 раза, будет некрасиво, уменьшим обратно
        if ((float)new_width / (float)new_height > 2.0)
        {
            new_width = src_width * 2;
            new_height = src_height * 2;
        }

        // Интерполируем
        dim3 dimGrid(
            src_width / 8 + (src_width % 8 ? 1 : 0),
            src_height / 8 + (src_height % 8 ? 1 : 0)
        );
        dim3 dimBlock(8, 8);
        cudaMemcpy(dev_im1data, imlib_image_get_data_for_reading_only(), src_width * src_height * sizeof(DATA32), cudaMemcpyHostToDevice);
        bicubicInterpolation<<< dimGrid, dimBlock >>>(dev_im1data, src_width, src_height, dev_im2data, new_width, new_height);
        imlib_free_image_and_decache();
        cudaThreadSynchronize();
        
        // Дополним до размера экрана
        padImage<<< 1, 1 >>>(dev_im2data, new_width, new_height, dev_im1data, screen->width, screen->height);
        // Создадим new_image
        DATA32* new_image_data = (DATA32*)malloc(screen->width * screen->height * sizeof(DATA32));
        cudaMemcpy(new_image_data, dev_im1data, screen->width * screen->height * sizeof(DATA32), cudaMemcpyDeviceToHost);
        new_image = imlib_create_image_using_data(screen->width, screen->height, new_image_data);
    }
    else
    {
        new_image = default_image;
        imlib_context_set_image(new_image);
        cudaMemcpy(dev_im1data, imlib_image_get_data_for_reading_only(), imlib_image_get_width() * imlib_image_get_height() * sizeof(DATA32), cudaMemcpyHostToDevice);
    }

    // Построим переходы
    for (int i = 0; i < 25; i++)
    {
        mixImages<<< screen->width * screen->height / 64, 64 >>>(dev_im1data, dev_im3data, (i + 1) * 0.04, dev_im2data);
        cudaThreadSynchronize();
        cudaMemcpy(transitionImageData[i], dev_im2data, screen->width * screen->height * sizeof(DATA32), cudaMemcpyDeviceToHost);
    }
    
    // Нарисуем переходы
    for (int i = 0; i < 25; i++)
    {
        imlib_context_set_image(transitionImage[i]);
        imlib_render_image_on_drawable(0, 0);
        XSetWindowBackgroundPixmap(display, root, pixmap);
        XClearWindow(display, root);
        XUngrabServer(display);
        XFlush(display);
    }

    // В 3 перед началом следующего вызова будет текущее изображение
    DATA32* tmp = dev_im3data;
    dev_im3data = dev_im2data;
    dev_im2data = tmp;

    // Освободим память
    if (new_image != default_image)
    {
        imlib_context_set_image(new_image);
        free(imlib_image_get_data_for_reading_only());
        imlib_free_image_and_decache();
    }
}

int main(int argc, char* argv[])
{
    // CUDA
    cudaSetDevice(0);
    
    // X11
    display = XOpenDisplay(NULL);
    root = RootWindow(display, DefaultScreen(display));
    screen = ScreenOfDisplay(display, DefaultScreen(display));
    pixmap = XCreatePixmap(display, root, screen->width, screen->height, DefaultDepth(display, DefaultScreen(display)));

    // ImLib
    imlib_context_set_display(display);
    imlib_context_set_drawable(pixmap);
    imlib_context_set_visual(DefaultVisual(display, DefaultScreen(display)));

    // MPD
    mpd_thread_data_t mpd_thread_data;
    mpd_thread_data.timeout = 30000;
    mpd_thread_data.poll_interval = 1;
    pthread_mutex_init(&mpd_thread_data.mutex, NULL);
    
    pthread_t mpd_thread;
    pthread_create(&mpd_thread, NULL, mpd_thread_body, (void*)&mpd_thread_data);

    // Last.fm
    lastfm_thread_data_t lastfm_thread_data;
    lastfm_thread_data.mpd = &mpd_thread_data;
    lastfm_thread_data.directory = "/home/themylogin/theMediaCenter/MpdWallpaperChanger/cache";
    lastfm_thread_data.poll_interval = 1;
    lastfm_thread_data.expire_interval = 86400;
    
    pthread_t lastfm_thread;
    pthread_create(&lastfm_thread, NULL, lastfm_thread_body, (void*)&lastfm_thread_data);
    
    // Memory
    cudaMalloc((void**)&dev_im1data, 3172          * 3172           * sizeof(DATA32));
    cudaMalloc((void**)&dev_im2data, screen->width * screen->height * sizeof(DATA32));
    cudaMalloc((void**)&dev_im3data, screen->width * screen->height * sizeof(DATA32));
    for (int i = 0; i < 25; i++)
    {
        transitionImageData[i] = (DATA32*)malloc(screen->width * screen->height * sizeof(DATA32));
        transitionImage[i] = imlib_create_image_using_data(screen->width, screen->height, transitionImageData[i]);
    }

    // Images
    default_image = imlib_load_image_immediately_without_cache("/home/themylogin/Wallpapers/01300_highwayatnightreload_1920x1080.jpg");
    imlib_context_set_image(default_image);
    cudaMemcpy(dev_im3data, imlib_image_get_data_for_reading_only(), imlib_image_get_width() * imlib_image_get_height() * sizeof(DATA32), cudaMemcpyHostToDevice);
    
    // Main loop
    char current_artist[1024];
    while (1)
    {
        pthread_mutex_lock(&mpd_thread_data.mutex);
        if (mpd_thread_data.is_playing)
        {
            strcpy(current_artist, mpd_thread_data.now_playing_artist);
            pthread_mutex_unlock(&mpd_thread_data.mutex);

            char* current_artist_dir = (char*)malloc(strlen(lastfm_thread_data.directory) + 1 + strlen(current_artist) + 1);
            sprintf(current_artist_dir, "%s/%s", lastfm_thread_data.directory, current_artist);
            DIR* dp = opendir(current_artist_dir);
            if (dp)
            {
                startDir:
                struct dirent* de;
                int read_something = 0;
                while (de = readdir(dp))
                {
                    if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, ".."))
                    {
                        continue;
                    }
                    read_something = 1;

                    char* image = (char*)malloc(strlen(current_artist_dir) + 1 + strlen(de->d_name) + 1);
                    sprintf(image, "%s/%s", current_artist_dir, de->d_name);
                    set_image(image);
                    free(image);

                    for (int i = 0; i < 18; i++)
                    {
                        sleep(1);

                        // Проверим, вдруг музыка кончилась или сменилась на другую
                        pthread_mutex_lock(&mpd_thread_data.mutex);
                        if (!mpd_thread_data.is_playing || strcmp(current_artist, mpd_thread_data.now_playing_artist))
                        {
                            pthread_mutex_unlock(&mpd_thread_data.mutex);
                            goto closeDir;
                        }
                        pthread_mutex_unlock(&mpd_thread_data.mutex);
                    }
                }

                if (!read_something)
                {
                    sleep(1);
                }
                rewinddir(dp);
                goto startDir;

                closeDir:
                closedir(dp);
            }
            free(current_artist_dir);
        }
        else
        {
            pthread_mutex_unlock(&mpd_thread_data.mutex);
            set_image(NULL);
        }

        sleep(1);
    }
}
