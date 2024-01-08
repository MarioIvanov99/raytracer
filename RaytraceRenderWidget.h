//////////////////////////////////////////////////////////////////////
//
//	University of Leeds
//	COMP 5812M Foundations of Modelling & Rendering
//	User Interface for Coursework
//
//	September, 2020
//
//  -----------------------------
//  Raytrace Render Widget
//  -----------------------------
//
//	Provides a widget that displays a fixed image
//	Assumes that the image will be edited (somehow) when Render() is called
//
////////////////////////////////////////////////////////////////////////

// include guard
#ifndef _RAYTRACE_RENDER_WIDGET_H
#define _RAYTRACE_RENDER_WIDGET_H

#define GAMMA 2.2
#define ALBEDOTHRESHOLD 0.0001
#define N_SAMPLES 500.0

// include the relevant QT headers
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QVector2D>
#include <QVector3D>
#include <QRandomGenerator>
#include <random>

// and include all of our own headers that we need
#include "TexturedObject.h"
#include "RenderParameters.h"


	// Surfel class
class Surfel
    { // Surfel class
	   public:
 // Surfel attributes
		unsigned int triangle; //Which triangle is this surfel part of
		Cartesian3 position; //The position. Position of pixel + interpolated depth
		Cartesian3 uv; //Texture coordinates
		Cartesian3 normal; //Interpolated normal
		Cartesian3 emission; //Emissive color of material
		Cartesian3 AmbientAlbedo; //Ambient color of material
		Cartesian3 LambertAlbedo; //Diffuse color of material
		Cartesian3 GlossyAlbedo; //Specular color of material
		float GlossyExponent; //Specular exponent
    int texmat; //Which texture and material to use for surfel
    float impulse;
    Cartesian3 ImpulseAlbedo;

		Cartesian3 BRDF(Cartesian3, Cartesian3); // BRDF function

	  }; // Surfel class

    // Light class
class Light
    { // Light class
  	public:
   // Light attributes
  	Homogeneous4 position;
    Cartesian3 intensity; //Light color. Just white for the most part

  }; // Light class



// class for a render widget with arcball linked to an external arcball widget
class RaytraceRenderWidget : public QOpenGLWidget
	{ // class RaytraceRenderWidget
	Q_OBJECT
	private:
	// the geometric object to be rendered
	TexturedObject *texturedObject;

	// the render parameters to use
	RenderParameters *renderParameters;

	// An image to use as a framebuffer
	RGBAImage frameBuffer;

  // A vector for all lights
  std::vector<Light> Luminaire;

  // The location of the eye
  Cartesian3 eye;

  // The final color, in float
  std::vector<std::vector<Cartesian3>> realColor;

  // Flag for origin rays
  bool isOrigin;

  std::uniform_real_distribution<> dist;

	public:
	// constructor
	RaytraceRenderWidget
			(
	 		// the geometric object to show
			TexturedObject 		*newTexturedObject,
			// the render parameters to use
			RenderParameters 	*newRenderParameters,
			// parent widget in visual hierarchy
			QWidget 			*parent
			);

	// destructor
	~RaytraceRenderWidget();

	protected:
	// called when OpenGL context is set up
	void initializeGL();
	// called every time the widget is resized
	void resizeGL(int w, int h);
	// called every time the widget needs painting
	void paintGL();

  // Gets the ortho matrix;
  Matrix4 Ortho(float, float, float, float, float, float);
  //Determines if point is inside a triangle defined by three points
  Homogeneous4 IsInside(Cartesian3, Cartesian3, Cartesian3, Cartesian3);
  // The current texture
  RGBAImage texture;

  // routine that generates the image
  void Raytrace();
	// routine that calculates the brdf for surfels
  float getBRDF(Cartesian3, Cartesian3);
	// routine that gets pixel color;
  Cartesian3 pathTrace(Cartesian3, Cartesian3, Cartesian3);
  // routine that finds the closest triangle along a ray;
  Homogeneous4 ClosestTriangleAlong(Cartesian3, Cartesian3);
  // routine that sets the parameters of the surfel
  Surfel Intersection(Homogeneous4);
  // routine that returns the directlight color
  Cartesian3 DirectLight(Surfel, Cartesian3, Light);
  // routine that returns the directlight color
  Cartesian3 IndirectLight(Surfel, Cartesian3, Cartesian3);
  // routine that does gamma correction and clamping;
  Cartesian3 GammaCorrection(Cartesian3);
  // routine for finding a random direction
  Cartesian3 MonteCarloHemisphereVector(Cartesian3);
  // get partial framebuffer
  void drawPartialImage(bool);


	// mouse-handling
	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseMoveEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	void keyPressEvent(QKeyEvent *event);

	public:
  // Copy of the vertices
	std::vector<Cartesian3> verticesCopy;
  // Copy of the vertices copy. Used for intersection
	std::vector<Cartesian3> verticesCopy2;
  // Copy of the normals
  std::vector<Cartesian3> normalsCopy;

	// these signals are needed to support shared arcball control
	public:
	signals:
	// these are general purpose signals, which scale the drag to
	// the notional unit sphere and pass it to the controller for handling
	void BeginScaledDrag(int whichButton, float x, float y);
	// note that Continue & End assume the button has already been set
	void ContinueScaledDrag(float x, float y);
	void EndScaledDrag(float x, float y);
	}; // class RaytraceRenderWidget

#endif
