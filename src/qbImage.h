/* ***********************************************************
	qbImage.hpp

	The qbImage class definition - A simple class for 2D
	image handling.

	This file forms part of the qbRayTrace project as described
	in the series of videos on the QuantitativeBytes YouTube
	channel.

	This code corresponds specifically to Episode 6 of the series,
	which may be found here:
	https://youtu.be/9K9ZYq6KgFY

	The whole series may be found on the QuantitativeBytes
	YouTube channel at:
	www.youtube.com/c/QuantitativeBytes

	GPLv3 LICENSE
	Copyright (c) 2021 Michael Bennett

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.

***********************************************************/

#ifndef QBIMAGE_H
#define QBIMAGE_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "SDL2/include/SDL.h"
class qbImage
{
public:
	// The constructor.
	__host__ __device__ qbImage();

	// The destructor.
	__host__ __device__ ~qbImage();

	// Function to initialise.
	__host__ __device__ void Initialize(const int xSize, const int ySize, SDL_Renderer* pRenderer);

	// Function to set the colour of a pixel.
	__host__ __device__ void SetPixel(const int x, const int y, const double red, const double green, const double blue);

	// Function to return the image for display.
	void Display();

	// Functions to return the dimensions of the image.
	int GetXSize();
	int GetYSize();

private:
	Uint32 ConvertColor(const double red, const double green, const double blue);
	void InitTexture();
	void ComputeMaxValues();

public:
	// Arrays to store image data.
	// std::vector<std::vector<double>> m_rChannel;
	// std::vector<std::vector<double>> m_gChannel;
	// std::vector<std::vector<double>> m_bChannel;
	float* m_rChannel;
	float* m_gChannel;
	float* m_bChannel;

	// Store the dimensions of the image.
	int m_xSize, m_ySize;

	// Store the maximum values.
	double m_maxRed, m_maxGreen, m_maxBlue, m_overallMax;

	// SDL2 stuff.
	SDL_Renderer* m_pRenderer;
	SDL_Texture* m_pTexture;

};

#endif

















