/* ***********************************************************
	qbImage.cpp

	The qbImage class implementation - A simple class for 2D
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

#include "qbImage.h"

// The default constructor.
__host__ __device__ qbImage::qbImage()
{
	m_xSize = 0;
	m_ySize = 0;
	m_pTexture = NULL;
}

// The destructor.
__host__ __device__ qbImage::~qbImage()
{
	if (m_pTexture != NULL)
		SDL_DestroyTexture(m_pTexture);
}

// Function to inialize.
__host__ __device__ void qbImage::Initialize(const int xSize, const int ySize, SDL_Renderer* pRenderer)
{
	// Resize the image arrays.
	// m_rChannel.resize(xSize, std::vector<double>(ySize, 0.0));
	// m_gChannel.resize(xSize, std::vector<double>(ySize, 0.0));
	// m_bChannel.resize(xSize, std::vector<double>(ySize, 0.0));
	m_rChannel = new float[xSize * ySize];
	m_gChannel = new float[xSize * ySize];
	m_bChannel = new float[xSize * ySize];

	// Store the dimensions.
	m_xSize = xSize;
	m_ySize = ySize;

	// Store the pointer to the renderer.
	m_pRenderer = pRenderer;

	// Initialise the texture.
	InitTexture();
}

// Function to set pixels.
__host__ __device__ void qbImage::SetPixel(const int x, const int y, const double red, const double green, const double blue)
{
	// m_rChannel.at(x).at(y) = red;
	// m_gChannel.at(x).at(y) = green;
	// m_bChannel.at(x).at(y) = blue;
	m_rChannel[(y * m_xSize) + x] = red;
	m_gChannel[(y * m_xSize) + x] = green;
	m_bChannel[(y * m_xSize) + x] = blue;
}

// Function to return the dimensions of the image.
int qbImage::GetXSize()
{
	return m_xSize;
}
int qbImage::GetYSize()
{
	return m_ySize;
}

// Function to generate the display.
void qbImage::Display()
{
	// Compute maximum values.
	ComputeMaxValues();

	// Allocate memory for a pixel buffer.
	Uint32* tempPixels = new Uint32[m_xSize * m_ySize];

	// Clear the pixel buffer.
	memset(tempPixels, 0, m_xSize * m_ySize * sizeof(Uint32));

	for (int x = 0; x < m_xSize; ++x)
	{
		for (int y = 0; y < m_ySize; ++y)
		{
			// tempPixels[(y * m_xSize) + x] = ConvertColor(m_rChannel.at(x).at(y), m_gChannel.at(x).at(y), m_bChannel.at(x).at(y));
			tempPixels[(y * m_xSize) + x] = ConvertColor(m_rChannel[(y * m_xSize) + x], m_gChannel[(y * m_xSize) + x], m_bChannel[(y * m_xSize) + x]);
		}
	}

	// Update the texture with the pixel buffer.
	SDL_UpdateTexture(m_pTexture, NULL, tempPixels, m_xSize * sizeof(Uint32));

	// Delete the pixel buffer.
	delete[] tempPixels;

	// Copy the texture to the renderer.
	SDL_Rect srcRect, bounds;
	srcRect.x = 0;
	srcRect.y = 0;
	srcRect.w = m_xSize;
	srcRect.h = m_ySize;
	bounds = srcRect;
	SDL_RenderCopy(m_pRenderer, m_pTexture, &srcRect, &bounds);
}

// Function to initialize the texture.
void qbImage::InitTexture()
{
	// Initialize the texture.
	Uint32 rmask, gmask, bmask, amask;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	rmask = 0xff000000;
	gmask = 0x00ff0000;
	bmask = 0x0000ff00;
	amask = 0x000000ff;
#else
	rmask = 0x000000ff;
	gmask = 0x0000ff00;
	bmask = 0x00ff0000;
	amask = 0xff000000;
#endif

	// Delete any previously created texture.
	if (m_pTexture != NULL)
		SDL_DestroyTexture(m_pTexture);

	// Create the texture that will store the image.
	SDL_Surface* tempSurface = SDL_CreateRGBSurface(0, m_xSize, m_ySize, 32, rmask, gmask, bmask, amask);
	m_pTexture = SDL_CreateTextureFromSurface(m_pRenderer, tempSurface);
	SDL_FreeSurface(tempSurface);
}

// Function to convert colours to Uint32
Uint32 qbImage::ConvertColor(const double red, const double green, const double blue)
{
	// Convert the colours to unsigned integers.
	unsigned char r = static_cast<unsigned char>((red / m_overallMax) * 255.0);
	unsigned char g = static_cast<unsigned char>((green / m_overallMax) * 255.0);
	unsigned char b = static_cast<unsigned char>((blue / m_overallMax) * 255.0);

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	Uint32 pixelColor = (r << 24) + (g << 16) + (b << 8) + 255;
#else
	Uint32 pixelColor = (255 << 24) + (r << 16) + (g << 8) + b;
#endif

	return pixelColor;
}

// Function to compute maximum values.
void qbImage::ComputeMaxValues()
{
	m_maxRed = 0.0;
	m_maxGreen = 0.0;
	m_maxBlue = 0.0;
	m_overallMax = 0.0;
	for (int x = 0; x < m_xSize; ++x)
	{
		for (int y = 0; y < m_ySize; ++y)
		{
			double redValue = m_rChannel[(y * m_xSize) + x];
			double greenValue = m_gChannel[(y * m_xSize) + x];
			double blueValue = m_bChannel[(y * m_xSize) + x];

			if (redValue > m_maxRed)
				m_maxRed = redValue;

			if (greenValue > m_maxGreen)
				m_maxGreen = greenValue;

			if (blueValue > m_maxBlue)
				m_maxBlue = blueValue;

			if (m_maxRed > m_overallMax)
				m_overallMax = m_maxRed;

			if (m_maxGreen > m_overallMax)
				m_overallMax = m_maxGreen;

			if (m_maxBlue > m_overallMax)
				m_overallMax = m_maxBlue;
		}
	}
}





















