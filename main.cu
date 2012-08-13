#include <dirent.h>
#include <math.h>
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

Imlib_Font font;
Imlib_Image lyrics_image = NULL;

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
        if (src_width * src_height > 3172 * 3172)
        {
            imlib_free_image_and_decache();
            return;
        }

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
        if ((float)new_width / (float)src_width > 2.0)
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

    // Напишем на них на всех текст песни
    if (lyrics_image != NULL)
    {
        imlib_context_set_image(lyrics_image);
        int lyrics_width = imlib_image_get_width();
        int lyrics_height = imlib_image_get_height();
        for (int i = 0; i < 25; i++)
        {
            imlib_context_set_image(transitionImage[i]);
            imlib_blend_image_onto_image(lyrics_image, true, 0, 0, lyrics_width, lyrics_height, screen->width - lyrics_width, 0, lyrics_width, lyrics_height);
        }
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

Imlib_Image draw_lyrics(char lyrics[][54], int lines)
{
    int margin = 36;

    imlib_context_set_font(font);

    int column_width = 0;
    int column_height = 0;
    for (int i = 0; i < lines; i++)
    {
        int text_w, text_h;        
        imlib_get_text_size(lyrics[i], &text_w, &text_h);
        if (text_w > column_width)
        {
            column_width = text_w;
        }
        column_height += text_h;
    }
    int column_count = ceil((float) column_height / (screen->height - 2 * margin));

    Imlib_Image im = imlib_create_image(margin + (column_width + margin) * column_count, screen->height);
    imlib_context_set_image(im);
    imlib_image_set_has_alpha(true);
    imlib_context_set_color(0, 0, 0, 0);
    imlib_image_fill_rectangle(0, 0, margin + (column_width + margin) * column_count, screen->height);

    imlib_context_set_color(255, 255, 255, 255);
    int x = margin, y = margin;
    bool column_begin = true;
    for (int i = 0; i < lines; i++)
    {
        if (column_begin && !strcmp(lyrics[i], ""))
        {
            continue;
        }

        column_begin = false;
        int width_return, height_return, horizontal_advance_return, vertical_advance_return;
        imlib_text_draw_with_return_metrics(x, y, lyrics[i], &width_return, &height_return, &horizontal_advance_return, &vertical_advance_return);

        y += height_return;
        if (y > screen->height - margin)
        {
            x += column_width + margin;
            y = margin;
            column_begin = true;
        }
    }

    return im;
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

    imlib_add_path_to_font_path("/usr/share/fonts/TTF");
    font = imlib_load_font("calibrib/18");

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
    char current_lyrics[1024][54];
    int current_lyrics_lines;    
    char current_lyrics_filename[1024]; strcpy(current_lyrics_filename, "");
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
                        // Тексты песен
                        char nowplaying_lyrics_filename[1024];
                        sprintf(nowplaying_lyrics_filename, "%s/.lyrics/%s-%s.txt", getenv("HOME"), mpd_thread_data.now_playing_artist, mpd_thread_data.now_playing_track);
                        if (strcmp(current_lyrics_filename, nowplaying_lyrics_filename))
                        {
                            FILE* lyrics_fh = fopen(nowplaying_lyrics_filename, "r");
                            if (lyrics_fh)
                            {
                                current_lyrics_lines = 0;
                                int next_line_offset = 0;
                                while (!feof(lyrics_fh))
                                {
                                    fgets(current_lyrics[current_lyrics_lines] + next_line_offset, 54 - next_line_offset, lyrics_fh);
                                    if (current_lyrics[current_lyrics_lines][strlen(current_lyrics[current_lyrics_lines]) - 1] != '\n' && !feof(lyrics_fh))
                                    {
                                        // Строка разорвалась не по своей воле
                                        int pos;
                                        for (pos = strlen(current_lyrics[current_lyrics_lines]) - 1; pos > 0; pos--)
                                        {
                                            if (current_lyrics[current_lyrics_lines][pos] == ' ')
                                            {
                                                break;
                                            }
                                        }
                                        if (pos != 0)
                                        {
                                            strcpy(current_lyrics[current_lyrics_lines + 1], current_lyrics[current_lyrics_lines] + pos + 1);
                                            next_line_offset = strlen(current_lyrics[current_lyrics_lines + 1]);
                                            current_lyrics[current_lyrics_lines][pos] = '\0';
                                        }
                                        else
                                        {
                                            next_line_offset = 0;
                                        }
                                    }
                                    else
                                    {
                                        current_lyrics[current_lyrics_lines][strlen(current_lyrics[current_lyrics_lines]) - 1] = '\0';
                                        next_line_offset = 0;
                                    }

                                    current_lyrics_lines++;
                                }
                                fclose(lyrics_fh);
                                if (next_line_offset)
                                {
                                    current_lyrics_lines++;
                                }

                                if (lyrics_image != NULL)
                                {
                                    imlib_context_set_image(lyrics_image);
                                    imlib_free_image_and_decache();
                                }
                                char lyrics_image_filename[1024];
                                sprintf(lyrics_image_filename, "/home/themylogin/theMediaCenter/MpdWallpaperChanger/lyrics/%s-%s.png", mpd_thread_data.now_playing_artist, mpd_thread_data.now_playing_track);
                                if ((lyrics_image = imlib_load_image_immediately_without_cache(lyrics_image_filename)) == NULL)
                                {
                                    lyrics_image = draw_lyrics(current_lyrics, current_lyrics_lines);
                                    imlib_context_set_image(lyrics_image);
                                    imlib_save_image(lyrics_image_filename);
                                }
                                strcpy(current_lyrics_filename, nowplaying_lyrics_filename);
                            }
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
