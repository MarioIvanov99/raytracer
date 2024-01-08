//////////////////////////////////////////////////////////////////////
//
//  University of Leeds
//  COMP 5812M Foundations of Modelling & Rendering
//  User Interface for Coursework
//
//  September, 2020
//
//  -----------------------------
//  Raytrace Render Widget
//  -----------------------------
//
//	Provides a widget that displays a fixed image
//	Assumes that the image will be edited (somehow) when Render() is called
//
////////////////////////////////////////////////////////////////////////

#include <math.h>

// include the header file
#include "RaytraceRenderWidget.h"

// constructor
RaytraceRenderWidget::RaytraceRenderWidget
        (
        // the geometric object to show
        TexturedObject      *newTexturedObject,
        // the render parameters to use
        RenderParameters    *newRenderParameters,
        // parent widget in visual hierarchy
        QWidget             *parent
        )
    // the : indicates variable instantiation rather than arbitrary code
    // it is considered good style to use it where possible
    :
    // start by calling inherited constructor with parent widget's pointer
    QOpenGLWidget(parent),
    // then store the pointers that were passed in
    texturedObject(newTexturedObject),
    renderParameters(newRenderParameters),
    dist(0.0, 1.0)
    { // constructor
    // leaves nothing to put into the constructor body
    setFocusPolicy(Qt::StrongFocus);

    } // constructor

// destructor
RaytraceRenderWidget::~RaytraceRenderWidget()
    { // destructor
    // empty (for now)
    // all of our pointers are to data owned by another class
    // so we have no responsibility for destruction
    // and OpenGL cleanup is taken care of by Qt
    } // destructor

// called when OpenGL context is set up
void RaytraceRenderWidget::initializeGL()
    { // RaytraceRenderWidget::initializeGL()
	// this should remain empty
    } // RaytraceRenderWidget::initializeGL()

// called every time the widget is resized
void RaytraceRenderWidget::resizeGL(int w, int h)
    { // RaytraceRenderWidget::resizeGL()
    // resize the render image
    frameBuffer.Resize(w, h);
    realColor.resize(frameBuffer.height);
    for(int i = 0; i < realColor.size(); i++){
      realColor[i].resize(frameBuffer.width);
    }
    } // RaytraceRenderWidget::resizeGL()

// called every time the widget needs painting
void RaytraceRenderWidget::paintGL()
    { // RaytraceRenderWidget::paintGL()
    // set background colour to white
    glClearColor(1.0, 0.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    setFocusPolicy(Qt::StrongFocus);
    // and display the image
    glDrawPixels(frameBuffer.width, frameBuffer.height, GL_RGBA, GL_UNSIGNED_BYTE, frameBuffer.block);
    } // RaytraceRenderWidget::paintGL()

Matrix4 RaytraceRenderWidget::Ortho(float left, float right, float bottom, float top, float near, float far){

  Matrix4 orthom;
  orthom[0][0] = 2/(right-left);
  orthom[0][3] = -(right+left)/(right-left);
  orthom[1][1] = 2/(top-bottom);
  orthom[1][3] = -(top+bottom)/(top-bottom);
  orthom[2][2] = -2/(far-near); // Mixing far-near and near-far. Is probably wrong, but textures are correct
  orthom[2][3] = -(near+far)/(near-far); // Currently do not know how to make textures work with both of them being
  orthom[3][3] = 1.0;                    // far-near or near-far

  return orthom;

}

//Determines if point is inside a triangle defined by three points
Homogeneous4 RaytraceRenderWidget::IsInside(Cartesian3 vertex0, Cartesian3 vertex1, Cartesian3 vertex2, Cartesian3 point){

  Homogeneous4 intertri;

  // now for each side of the triangle, compute the line vectors
  Cartesian3 vector01 = vertex1 - vertex0;
  Cartesian3 vector12 = vertex2 - vertex1;
  Cartesian3 vector20 = vertex0 - vertex2;

  // now compute the line normal vectors
  Cartesian3 normal01(-vector01.y, vector01.x, 0.0);
  Cartesian3 normal12(-vector12.y, vector12.x, 0.0);
  Cartesian3 normal20(-vector20.y, vector20.x, 0.0);

  // we don't need to normalise them, because the square roots will cancel out in the barycentric coordinates
  float lineConstant01 = normal01.dot(vertex0);
  float lineConstant12 = normal12.dot(vertex1);
  float lineConstant20 = normal20.dot(vertex2);

  // and compute the distance of each vertex from the opposing side
  float distance0 = normal12.dot(vertex0) - lineConstant12;
  float distance1 = normal20.dot(vertex1) - lineConstant20;
  float distance2 = normal01.dot(vertex2) - lineConstant01;

  // if any of these are zero, we will have a divide by zero error
  // but notice that if they are zero, the vertices are collinear in projection and the triangle is edge on
  // we can render that as a line, but the better solution is to render nothing.  In a surface, the adjacent
  // triangles will eventually take care of it
  if ((distance0 == 0) || (distance1 == 0) || (distance2 == 0)){

    intertri.w = -1;
    return intertri;

  }

  // right - we have a pixel inside the frame buffer AND the bounding box
  // note we *COULD* compute gamma = 1.0 - alpha - beta instead
  float alpha = (normal12.dot(point) - lineConstant12) / distance0;
  float beta = (normal20.dot(point) - lineConstant20) / distance1;
  float gamma = (normal01.dot(point) - lineConstant01) / distance2;

  // now perform the half-plane test
  // -0.00019 to remove black spots on edges. Does not fix them completely.
  if((alpha < -0.00019) || (beta < -0.00019) || (gamma < -0.00019)){

    intertri.w = -1;
    return intertri;

  }
  else{

    intertri.x = alpha;
    intertri.y = beta;
    intertri.z = gamma;
    intertri.w = 1;
    return intertri; //return a Homogenous4 with interpolation, if a triangle is found

  }

}

Homogeneous4 RaytraceRenderWidget::ClosestTriangleAlong(Cartesian3 origin, Cartesian3 direction){

  float z = 2147483647; // Max int
  Homogeneous4 triangle;
  float a=0, b=0, c=0;
  int doubleu = -1;
  Cartesian3 p, q, r, u, v, n, w, l, o;

  for(int i = 0; i<texturedObject->faceVertices.size(); i++){ // For every triangle

    //Initial ray uses DCS.
    //The reason for this is because when I use VCS for initial rays
    //Any change in z changes the position of the object on screen
    //Works correctly when x and y are in DCS.
    p = verticesCopy2[texturedObject->faceVertices[i][0]];
    q = verticesCopy2[texturedObject->faceVertices[i][1]];
    r = verticesCopy2[texturedObject->faceVertices[i][2]];

    //All other rays are in the same coordinates as the light
    if(origin.z != 0.0){

      p = verticesCopy[texturedObject->faceVertices[i][0]];
      q = verticesCopy[texturedObject->faceVertices[i][1]];
      r = verticesCopy[texturedObject->faceVertices[i][2]];

    }

    //Converting to planar coordinates
    u = (q-p).unit();
    v = (r-p).unit();
    n = u.cross(v).unit();
    w = n.cross(u).unit();
    l = direction.unit();
    if((p-origin).dot(n)/l.dot(n)<0) //Discard unwanted intersections
      continue;
    o = origin + l*(p-origin).dot(n)/l.dot(n);

    Cartesian3 op((o-p).dot(u),(o-p).dot(w),(o-p).dot(n));
    Cartesian3 pp((p-p).dot(u),(p-p).dot(w),(p-p).dot(n));
    Cartesian3 qp((q-p).dot(u),(q-p).dot(w),(q-p).dot(n));
    Cartesian3 rp((r-p).dot(u),(r-p).dot(w),(r-p).dot(n));

    triangle = IsInside(pp, qp, rp, op);

    if(triangle.w > -1){ //If there is an itnersection
      if((o-origin).length()<z){ //If the depth is smaller
        a = triangle.x; //Set the values of the Homogeneous4
        b = triangle.y;
        c = triangle.z;
        doubleu = i;
        z = (o-origin).length(); //Update current depth
      }
    }

  }

  triangle.x = a; //Set final values
  triangle.y = b;
  triangle.z = c;
  triangle.w = doubleu;
  return triangle;

}

Surfel RaytraceRenderWidget::Intersection(Homogeneous4 triangle){ //Set surfel attributes

  Surfel S;
  std::vector<unsigned int> vface = texturedObject->faceVertices[triangle.w]; //The first triangle the ray intersects
  std::vector<unsigned int> nface = texturedObject->faceNormals[triangle.w];
  std::vector<unsigned int> tface = texturedObject->faceTexCoords[triangle.w];

  //triangle x,y and z are alpha beta and gamma;
  //Only z needs to be interpolated for position
  S.position.z = verticesCopy[vface[0]].z*triangle.x + verticesCopy[vface[1]].z*triangle.y + verticesCopy[vface[2]].z*triangle.z;
  S.normal.x = normalsCopy[nface[0]].x*triangle.x + normalsCopy[nface[1]].x*triangle.y + normalsCopy[nface[2]].x*triangle.z;
  S.normal.y = normalsCopy[nface[0]].y*triangle.x + normalsCopy[nface[1]].y*triangle.y + normalsCopy[nface[2]].y*triangle.z;
  S.normal.z = normalsCopy[nface[0]].z*triangle.x + normalsCopy[nface[1]].z*triangle.y + normalsCopy[nface[2]].z*triangle.z;

  S.normal = S.normal.unit(); //Normals need to be converted to unit vectors

  //These lines are extremely long, which could be considered poor style
  //However, texture lookup with a more readable line would require copying
  //a texture on every pixel.
  //The textures provided are made up of 131072 RGBAvalues
  //This would mean that for every pixel, that many RGBAvalues
  //would have to be copied over, which severly slows down rendering
  S.uv.x = (texturedObject->textureCoords[tface[0]].x*triangle.x + texturedObject->textureCoords[tface[1]].x*triangle.y + texturedObject->textureCoords[tface[2]].x*triangle.z)*texturedObject->textures[texturedObject->whichTexMat[triangle.w]].width;
  S.uv.y = (texturedObject->textureCoords[tface[0]].y*triangle.x + texturedObject->textureCoords[tface[1]].y*triangle.y + texturedObject->textureCoords[tface[2]].y*triangle.z)*texturedObject->textures[texturedObject->whichTexMat[triangle.w]].height;

  //Get material properties
  S.emission.x = renderParameters->emissive;
  S.emission.y = renderParameters->emissive;
  S.emission.z = renderParameters->emissive;
  //S.AmbientAlbedo.x = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][0].x;
  //S.AmbientAlbedo.y = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][0].y;
  //S.AmbientAlbedo.z = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][0].z;
  S.LambertAlbedo.x = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][1].x;
  S.LambertAlbedo.y = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][1].y;
  S.LambertAlbedo.z = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][1].z;
  S.GlossyAlbedo.x = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][2].x;
  S.GlossyAlbedo.y = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][2].y;
  S.GlossyAlbedo.z = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][2].z;
  S.GlossyExponent = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][3].x;
  S.impulse = texturedObject->materials[texturedObject->whichTexMat[triangle.w]][3].y;
  S.ImpulseAlbedo.x = 0.9;
  S.ImpulseAlbedo.y = 0.9;
  S.ImpulseAlbedo.z = 0.9;

  S.texmat = triangle.w;

  return S;

}

Cartesian3 Surfel::BRDF(Cartesian3 InDirection, Cartesian3 OutDirection){ //Get brdf

  float L = normal.dot(InDirection)/(normal.length()*InDirection.length()); //Exactly as described in the slides
  Cartesian3 bisection = (OutDirection+InDirection)/2.0;
  float G = pow(normal.dot(bisection)/(normal.length()*bisection.length()), GlossyExponent);

  return L*LambertAlbedo + G*GlossyAlbedo;

}

Cartesian3 RaytraceRenderWidget::DirectLight(Surfel S, Cartesian3 OutDirection, Light light){ //Calculate DirectLight

  Cartesian3 InDirection;
  Cartesian3 noColor;
  Cartesian3 outColor;

  float dsqr = 1.0; //Attenuation
  Cartesian3 CheckShadow = (light.position.Vector() - S.position); //Shadow direction
  if(light.position.w == 0) //If directional light
    CheckShadow = light.position.Vector().reverse();
  Cartesian3 offset = S.position + 0.0002*CheckShadow; //To prevent shadow acne
  Homogeneous4 triangle = ClosestTriangleAlong(offset, CheckShadow);
  if(triangle.w != -1) // Return no color if in shadow
    return noColor;

  if(light.position.w == 0) //No attenuation if directional light
    InDirection = light.position.Vector();
  else{
    InDirection = S.position - light.position.Vector();
    dsqr = InDirection.dot(InDirection);
  }

  int u = S.uv.x; //Get texture color here
  int v = S.uv.y;

  Cartesian3 texColor;
  texColor.x = texturedObject->textures[texturedObject->whichTexMat[S.texmat]][v][u].red/255.0;
  texColor.y = texturedObject->textures[texturedObject->whichTexMat[S.texmat]][v][u].green/255.0;
  texColor.z = texturedObject->textures[texturedObject->whichTexMat[S.texmat]][v][u].blue/255.0;

  outColor = S.BRDF(InDirection, OutDirection); //modulate
  outColor.x = texColor.x*light.intensity.x*outColor.x/dsqr;
  outColor.y = texColor.y*light.intensity.y*outColor.y/dsqr;
  outColor.z = texColor.z*light.intensity.z*outColor.z/dsqr;



  return outColor;

}

Cartesian3 RaytraceRenderWidget::IndirectLight(Surfel S, Cartesian3 OutDirection, Cartesian3 combinedAlbedo){

  if(dist(*QRandomGenerator::global()) < 0.5)
    return eye; //eye is used for efficiency. It is already (0, 0, 0)
  Cartesian3 InDirection;
  Cartesian3 albedo;
  if(dist(*QRandomGenerator::global()) < S.impulse){ //As described in slides
    InDirection = 2.0*S.normal - OutDirection;
    albedo = S.ImpulseAlbedo; //ImpulseAlbedo is set to 0.9 for all
  }
  else{
    InDirection = MonteCarloHemisphereVector(S.normal);
    albedo = S.BRDF(InDirection, OutDirection);
  }
  Cartesian3 newAlbedo(albedo.x*combinedAlbedo.x, albedo.y*combinedAlbedo.y, albedo.z*combinedAlbedo.z);
  Cartesian3 InLight = pathTrace(S.position, InDirection, newAlbedo);
  Cartesian3 outColor(albedo.x*InLight.x, albedo.y*InLight.y, albedo.z*InLight.z);

  return outColor;

}

Cartesian3 RaytraceRenderWidget::pathTrace(Cartesian3 origin, Cartesian3 direction, Cartesian3 combinedAlbedo){

  if(combinedAlbedo.x < ALBEDOTHRESHOLD && combinedAlbedo.y < ALBEDOTHRESHOLD && combinedAlbedo.z < ALBEDOTHRESHOLD)
    return eye; //If all 3 are below the threshold

  Cartesian3 noColor;
  Cartesian3 totalRadiance;
  Cartesian3 OutDirection;
  Cartesian3 intorigin((origin.x+1)*(frameBuffer.width/2), (origin.y+1)*(frameBuffer.height/2), 0.0);
  Homogeneous4 triangle;

  if(isOrigin){ //Used so that initial intersection works correctly
    triangle = ClosestTriangleAlong(intorigin, direction); //Check for intersection
    isOrigin = false;
  }
  else
    triangle = ClosestTriangleAlong(origin, direction); //Check for intersection

  if(triangle.w>-1){ //If there is a triangle
    Surfel S = Intersection(triangle); //Set surfel
    S.position.x = origin.x;
    S.position.y = origin.y;

    for(int i = 0; i<Luminaire.size(); i++){ //For every light

      if(Luminaire[i].position.w == 0) //Towards eye or twoards point
        OutDirection = eye;
      else
        OutDirection = S.position;

      totalRadiance = totalRadiance + DirectLight(S, OutDirection, Luminaire[0]);

    }

    totalRadiance = totalRadiance + IndirectLight(S, OutDirection, combinedAlbedo); // Currently only one light anyway
    totalRadiance = S.emission + totalRadiance; //add emission at the end

    return totalRadiance;

  }

  return noColor;

}

Cartesian3 RaytraceRenderWidget::GammaCorrection(Cartesian3 color){ //Gamma correction

  Cartesian3 outColor; //GAMMA is 2.2
  outColor.x = pow(color.x, 1.0/GAMMA);
  outColor.y = pow(color.y, 1.0/GAMMA);
  outColor.z = pow(color.z, 1.0/GAMMA);

  //Clamping. Only required for when specular and diffuse are both very high
  if(outColor.x>1.0)
    outColor.x = 1.0;
  if(outColor.y>1.0)
    outColor.y = 1.0;
  if(outColor.z>1.0)
    outColor.z = 1.0;
  if(outColor.x<0.0)
    outColor.x = 0.0;
  if(outColor.y<0.0)
    outColor.y = 0.0;
  if(outColor.z<0.0)
    outColor.z = 0.0;

  return outColor;

}

// routine to get a random direction
Cartesian3 RaytraceRenderWidget::MonteCarloHemisphereVector(Cartesian3 normal){

  float u, v;
  Cartesian3 normalt;

  if(normal.x > normal.y){ //Get polar coordinates
    normalt.x = normal.z;
    normalt.z = -normal.x;
    normalt = normalt.unit();
  }
  else{
    normalt.y = -normal.z;
    normalt.z = normal.y;
    normalt = normalt.unit();
  }

  Cartesian3 normalb = normal.cross(normalt);

  u = dist(*QRandomGenerator::global()); //Get random numbers in range 0-1
  v = dist(*QRandomGenerator::global());

  Cartesian3 sample(cos(2*M_PI*u), v, sin(2*M_PI*u)); //Get sample in hemisphere
  Cartesian3 direction;
  direction.x = sample.x*normalb.x + sample.y*normal.x + sample.z*normalt.x; //Align hemisphere with normal
  direction.y = sample.x*normalb.y + sample.y*normal.y + sample.z*normalt.y;
  direction.z = sample.x*normalb.z + sample.y*normal.z + sample.z*normalt.z;

  return direction.unit();

}

void RaytraceRenderWidget::drawPartialImage(bool _final){ //Used for visualizing the image while it is rendering

  Cartesian3 gammaCorrectedColour;

  for(int i = 0; i < frameBuffer.width; i++){ //Ray for each pixel
    for(int j = 0; j < frameBuffer.height; j++){

      gammaCorrectedColour = GammaCorrection(realColor[j][i]); //True floating point accuracy
      frameBuffer[j][i].red = gammaCorrectedColour.x*255.0;
      frameBuffer[j][i].green = gammaCorrectedColour.y*255.0;
      frameBuffer[j][i].blue = gammaCorrectedColour.z*255.0;
      if(_final) //Reset the color to (0,0,0) if the entire image has been redered
        realColor[j][i] = eye;

    }
  }
  this->repaint();

}

    // routine that generates the image
void RaytraceRenderWidget::Raytrace()
    { // RaytraceRenderWidget::Raytrace()

      isOrigin = true; //Because I need the original rays to be in a different coordinate system. It is the only way initial intersection works correctly.

      // compute the aspect ratio of the widget
      float aspectRatio = (float) frameBuffer.width / (float) frameBuffer.height;

      //Projection matrix
      Matrix4 projectionM;

      //For scale button
      float scale = renderParameters->zoomScale;

      //For final color
      Cartesian3 gammaCorrectedColour;

      //For scale button
      if(renderParameters->scaleObject)
        scale/=texturedObject->objectSize;

      // The same operation as in rednerwidget
      if (aspectRatio > 1.0)
          projectionM = Ortho(-aspectRatio, aspectRatio, -1.0, 1.0, 0, 2.0);
      // otherwise, make left & right -1.0 & 1.0
      else
          projectionM = Ortho(-1.0, 1.0, -1.0/aspectRatio, 1.0/aspectRatio, 0, 2.0);


    verticesCopy = texturedObject->vertices; //Getting vertices
    verticesCopy2 = verticesCopy; //For DCS
    normalsCopy = texturedObject->normals; //Getting vertices

    Matrix4 modelview; //modelview No scaling is done in the modelview because the OpenGL side never calls glScalef
    modelview = renderParameters->rotationMatrix;
    modelview[0][3] = renderParameters->xTranslate;
    modelview[1][3] = renderParameters->yTranslate;
    modelview[2][3] = -1.0;

    Matrix4 modelviewl; //Same as modelview, but with lightMatrix instead
    modelviewl = renderParameters->lightMatrix;
    modelviewl[0][3] = renderParameters->xTranslate;
    modelviewl[1][3] = renderParameters->yTranslate;
    modelviewl[2][3] = -1.0;
    modelviewl[0][0] *= renderParameters->zoomScale;
    modelviewl[1][1] *= renderParameters->zoomScale;
    modelviewl[2][2] *= renderParameters->zoomScale;

    for(int i = 0; i < verticesCopy.size(); i++){ //Get VCS (and DCS)

      verticesCopy[i] = scale*verticesCopy[i]; // First scale
      if(renderParameters->centreObject){ //Then translate to centre. All given scenes are auto centred.

        verticesCopy[i].x += -texturedObject->centreOfGravity.x*scale;
        verticesCopy[i].y += -texturedObject->centreOfGravity.y*scale;
        verticesCopy[i].z += -texturedObject->centreOfGravity.z*scale;

      }
      //std::cout << "v " << verticesCopy[i] << '\n';
      verticesCopy[i] = modelview*verticesCopy[i]; //Modelview transformation
      verticesCopy[i] = projectionM*verticesCopy[i]; //Projection transformation
      verticesCopy2[i] = verticesCopy[i];
      verticesCopy2[i].x = (verticesCopy2[i].x+1)*(frameBuffer.width/2); //DCS transformation
      verticesCopy2[i].y = (verticesCopy2[i].y+1)*(frameBuffer.height/2);

    }

    for(int i = 0; i < normalsCopy.size(); i++){ //Normals just need rotation transformation

      normalsCopy[i] = renderParameters->rotationMatrix*normalsCopy[i]; //Multiplying them by the entire modelview matrix distorts them

    }

  	for(int i = 0; i < frameBuffer.width; i++){ //Set frameBufferto black
      for(int j = 0; j < frameBuffer.height; j++){
        frameBuffer[j][i].red = 0;
        frameBuffer[j][i].green = 0;
        frameBuffer[j][i].blue = 0;
      }
    }

    Light light1; //Set initial (point) light
    light1.position.x = renderParameters->lightPosition[0];
    light1.position.y = renderParameters->lightPosition[1];
    light1.position.z = renderParameters->lightPosition[2];
    light1.position.w = 1.0;


    light1.position = modelviewl*light1.position; //Transform

    light1.intensity.x = 1.0; //Set light color
    light1.intensity.y = 1.0;
    light1.intensity.z = 1.0;

    Luminaire.push_back(light1); //Put in lights vector
    float y = -1.0; //For getting VCS coordinates of each pixel
    float x = -1.0;

    for(int k = 0; k<N_SAMPLES; k++){
      y = -1.0;
      x = -1.0;
      for(int i = 0; i < frameBuffer.width; i++){ //Ray for each pixel
        for(int j = 0; j < frameBuffer.height; j++){
          Cartesian3 combinedAlbedo(1.0, 1.0, 1.0);
          Cartesian3 origin(y, x, 0.0);
          Cartesian3 direction(0.0, 0.0, 1.0);
          realColor[j][i] = realColor[j][i] + (1/N_SAMPLES)*pathTrace(origin, direction, combinedAlbedo);
          isOrigin = true;
          x += 2.0f/frameBuffer.height;
        }
        x = -1.0f;
        y += 2.0f/frameBuffer.width;
      }
      std::cout << k << '\n'; //Loading
      if(k%5 == 0)
      drawPartialImage(false); //So that something is getting rendered every 5 runs
    }

    drawPartialImage(true);

    Luminaire.clear(); //Clear vector after every raytrace call

    } // RaytraceRenderWidget::Raytrace()

// mouse-handling
void RaytraceRenderWidget::mousePressEvent(QMouseEvent *event)
    { // RaytraceRenderWidget::mousePressEvent()
    // store the button for future reference
    int whichButton = event->button();
    // scale the event to the nominal unit sphere in the widget:
    // find the minimum of height & width
    float size = (width() > height()) ? height() : width();
    // scale both coordinates from that
    float x = (2.0 * event->x() - size) / size;
    float y = (size - 2.0 * event->y() ) / size;


    // and we want to force mouse buttons to allow shift-click to be the same as right-click
    int modifiers = event->modifiers();

    // shift-click (any) counts as right click
    if (modifiers & Qt::ShiftModifier)
        whichButton = Qt::RightButton;

    // send signal to the controller for detailed processing
    emit BeginScaledDrag(whichButton, x,y);
    } // RaytraceRenderWidget::mousePressEvent()

void RaytraceRenderWidget::mouseMoveEvent(QMouseEvent *event)
    { // RaytraceRenderWidget::mouseMoveEvent()
    // scale the event to the nominal unit sphere in the widget:
    // find the minimum of height & width
    float size = (width() > height()) ? height() : width();
    // scale both coordinates from that
    float x = (2.0 * event->x() - size) / size;
    float y = (size - 2.0 * event->y() ) / size;

    // send signal to the controller for detailed processing
    emit ContinueScaledDrag(x,y);
    } // RaytraceRenderWidget::mouseMoveEvent()

void RaytraceRenderWidget::mouseReleaseEvent(QMouseEvent *event)
    { // RaytraceRenderWidget::mouseReleaseEvent()
    // scale the event to the nominal unit sphere in the widget:
    // find the minimum of height & width
    float size = (width() > height()) ? height() : width();
    // scale both coordinates from that
    float x = (2.0 * event->x() - size) / size;
    float y = (size - 2.0 * event->y() ) / size;

    // send signal to the controller for detailed processing
    emit EndScaledDrag(x,y);
    } // RaytraceRenderWidget::mouseReleaseEvent()

void RaytraceRenderWidget::keyPressEvent(QKeyEvent *event){

  if(event->key() == Qt::Key_R){
    Raytrace();
    this->repaint();
  }

}
