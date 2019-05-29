#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SDL/SDL.h>
#include <time.h>
#include <string.h>

#define ABS(a) ((a)>0?(a):-1*(a))

typedef struct
{
	double x,y,z;
} Vec3;

typedef struct
{
    double tx,ty,tz;
} Translation;

typedef struct
{
    double angle,ux,uy,uz;
} Rotation;

typedef struct
{
    double sx,sy,sz;
} Scaling;

typedef struct
{
	Vec3 a;
	Vec3 b;
	double u; /* texture coordinates of intersected point */
	double v;
} Ray;

typedef struct 
{
	Vec3 pos;
	Vec3 gaze;
	Vec3 v;
	Vec3 u;
	double l,r,b,t;
	double d;
} Camera;

typedef struct
{
	double r,g,b;
} Color;


typedef struct
{
	Color ambient;
	Color diffuse;
	Color specular;
	Color reflect;
	double specExp;

} Material;

typedef struct
{
	Color c;
	Vec3 pos;
} Light;

typedef struct
{
	int mat;
    int texture;
    int modelType; /* 0 cube, 1 sphere */
    int numTransformations;
    int transformationIDs[1000];
    char transformationTypes[1000];
    Vec3 center; // used only for sphere
    double radius; // used only for sphere
    Vec3 vertices[8]; // used only for the cube
} Model;


Camera cam;

int reflectCount = 0;

Light lights[100];
int numLights = 0;

int numModels = 0;
Model models[1000];

Material materials[1000];
int numMaterials;
int materialIDs[1000];

Translation translations[1000];
int numTranslations;

Rotation rotations[1000];
int numRotations;

Scaling scalings[1000];
int numScalings;

Color ambColor;

Color bgColor;

double dummy;

int sizeX, sizeY;

double pixelW, pixelH;

double halfPixelW, halfPixelH;

char outFileName[80];

Color **image;

int numTextures=0;
char textureNames[100][100];
Color ***texture;
int textWidth[100],textHeight[100];

void readTexture(char *tname, int tind)
{
        int max;
        int i,j;
        int r,g,b;
        int k;
        char ch;
        FILE *fp;

        fp = fopen(tname,"r");

        printf("texture filename = %s\n",tname);

        /*read the header*/

        fscanf(fp, "P%c\n", &ch);
        if (ch != '3') {
                fprintf(stderr, "Only ascii mode 3 channel PPM files");
                exit(-1);
        }

        /*strip comment lines*/
        ch = getc(fp);

	while (ch == '#') {
      		do {
                  	ch = getc(fp);
      		}
          	while (ch != '\n');
      		ch = getc(fp);
    	}
        ungetc(ch, fp);

        /*read the width*/
        fscanf(fp,"%d",&(textWidth[tind]));

        /*read the height*/
        fscanf(fp,"%d",&(textHeight[tind]));

        /*max intensity used to get texture color between 0 and 1*/
        fscanf(fp,"%d",&max);

        texture[tind] = (Color **)malloc(sizeof(Color *)*textHeight[tind]);
        for(i=0;i<textHeight[tind];++i){
                     texture[tind][i] = (Color *)malloc(sizeof(Color)*textWidth[tind]);
    	}

        for(i=0;i<textHeight[tind];++i){
                for(j=0;j<textWidth[tind];++j) {
                        fscanf(fp,"%d %d %d",&r,&g,&b);
                        texture[tind][i][j].r = r/(float)max;
                        texture[tind][i][j].g = g/(float)max;
                        texture[tind][i][j].b = b/(float)max;
                }
        }
        fclose(fp);

}

Vec3 cross(Vec3 a, Vec3 b)
{
	Vec3 tmp;
	
	tmp.x = a.y*b.z-b.y*a.z;
	tmp.y = b.x*a.z-a.x*b.z;
	tmp.z = a.x*b.y-b.x*a.y;
	
	return tmp;
}

double dot(Vec3 a, Vec3 b)
{
		return a.x*b.x+a.y*b.y+a.z*b.z;
}
int convert (double c)
{
/*	if (c>=1.0) return 255;
	if (c<=0.0) return 0;
	return (int)(255*c);*/
    if (c>=255.0) return 255;
    if (c<=0.0) return 0;
    return (int)(c);
}

double length2(Vec3 v)
{
	return (v.x*v.x+v.y*v.y+v.z*v.z);
}

double length(Vec3 v)
{
	return sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
}

Vec3 normalize(Vec3 v)
{
	Vec3 tmp;
	double d;
	
	d=length(v);
	tmp.x = v.x/d;
	tmp.y = v.y/d;
	tmp.z = v.z/d;
	
	return tmp;
}

Vec3 add(Vec3 a, Vec3 b)
{
	Vec3 tmp;
	tmp.x = a.x+b.x;
	tmp.y = a.y+b.y;
	tmp.z = a.z+b.z;
	
	return tmp;
}

Vec3 mult(Vec3 a, double c)
{
	Vec3 tmp;
	tmp.x = a.x*c;
	tmp.y = a.y*c;
	tmp.z = a.z*c;
	
	return tmp;
}

double distance(Vec3 a, Vec3 b)
{
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}

void printVec(Vec3 s)
{
	printf("(%lf,%lf,%lf)\n",s.x,s.y,s.z);
}

int equal(Vec3 a, Vec3 b)
{
	double e = 0.000000001;
	
	//printf("%lf %lf %f ----",ABS((a.x-b.x)),ABS((a.y-b.y)),ABS((a.z-b.z)));
	if ((ABS((a.x-b.x))<e) && (ABS((a.y-b.y))<e) && (ABS((a.z-b.z))<e))
		{ return 1;}
	else { return 0;}
}


//setup the putpixel function
void putpixel(SDL_Surface *surface, int x, int y, Uint32 pixel)
{
    int bpp = surface->format->BytesPerPixel;
    /* Here p is the address to the pixel we want to set */
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 1:
        *p = pixel;
        break;

    case 2:
        *(Uint16 *)p = pixel;
        break;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            p[0] = (pixel >> 16) & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = pixel & 0xff;
        } else {
            p[0] = pixel & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = (pixel >> 16) & 0xff;
        }
        break;

    case 4:
        *(Uint32 *)p = pixel;
        break;
    }
}

void readCamera(char *fName)
{
    FILE *f;
    int i;
    char line[80]={};
    f = fopen(fName,"r");
    
    fgets(line,80,f); // skip first 2 lines. this program reads only the first camera
    fgets(line,80,f);
    
    fscanf(f,"%lf %lf %lf",&cam.pos.x,&cam.pos.y,&cam.pos.z);
    fscanf(f,"%lf %lf %lf",&cam.gaze.x,&cam.gaze.y,&cam.gaze.z);
    fscanf(f,"%lf %lf %lf",&cam.v.x,&cam.v.y,&cam.v.z);
    
    cam.u = cross(cam.gaze,cam.v);
    
    cam.u = normalize(cam.u);
    
    cam.v = cross(cam.u,cam.gaze);
    cam.v = normalize(cam.v);
    
    cam.gaze = normalize(cam.gaze);
    
    fscanf(f,"%lf %lf %lf %lf",&cam.l,&cam.r,&cam.b,&cam.t);
    fscanf(f,"%lf",&cam.d);
    fscanf(f,"%d %d",&sizeX,&sizeY);
    
    fscanf(f,"%s",outFileName);
    
    // camera reading complete
    
    pixelW = (cam.r-cam.l)/(double)sizeX;
    halfPixelW = pixelW*0.5;
    
    pixelH = (cam.t-cam.b)/(double)sizeY;
    halfPixelH = pixelH*0.5;
    
    cam.gaze = normalize(cam.gaze);
    cam.v = normalize(cam.v);
    cam.u = cross(cam.v,mult(cam.gaze,-1));
    cam.u = normalize(cam.u);
    cam.v = cross(mult(cam.gaze,-1),cam.u);
    cam.v = normalize(cam.v);

    image = (Color **)malloc(sizeof(Color *)*sizeX);
    
    if (image==NULL)
    {
        printf("Cannot allocate memory for image.");
        exit(1);
    }
    
    for (i=0;i<sizeX;i++)
    {
        image[i] = (Color *)malloc(sizeof(Color)*sizeY);
        if (image[i]==NULL)
        {
            printf("Cannot allocate memory for image.");
            exit(1);
        }
    }
}


void readScene(char *fName)
{
    FILE *f;
    char line[80];
    int i,j,k,id;
    int numT,matID;
    char tmp[80];

    f = fopen(fName,"r");
    
    fscanf(f,"%d",&reflectCount);

    fscanf(f,"%lf %lf %lf",&(bgColor.r),&(bgColor.g),&(bgColor.b));

    fscanf(f,"%lf %lf %lf",&(ambColor.r),&(ambColor.g),&(ambColor.b));

    fscanf(f,"%d",&numLights);
    
    for (i=0;i<numLights;i++)
    {

        fscanf(f,"%lf %lf %lf",&(lights[i].pos.x),&(lights[i].pos.y),&(lights[i].pos.z));
        fscanf(f,"%lf %lf %lf",&(lights[i].c.r),&(lights[i].c.g),&(lights[i].c.b));
    
    }
    
    fscanf(f,"%d",&numMaterials);
    
    for (i=1;i<=numMaterials;i++)
    {
        fscanf(f,"%s",line); // skip word: #Material
        fscanf(f,"%d",&id); // skip word: material id

        fscanf(f,"%lf %lf %lf",&(materials[i].ambient.r),&(materials[i].ambient.g),&(materials[i].ambient.b));
        
        fscanf(f,"%lf %lf %lf",&(materials[i].diffuse.r),&(materials[i].diffuse.g),&(materials[i].diffuse.b));
        
        fscanf(f,"%lf %lf %lf %lf",&(materials[i].specular.r),&(materials[i].specular.g),&(materials[i].specular.b),&(materials[i].specExp));
        
        fscanf(f,"%lf %lf %lf",&(materials[i].reflect.r),&(materials[i].reflect.g),&(materials[i].reflect.b));
        
        materialIDs[i] = i+1;
        
    }
    
    printf("Num materials = %d\n",numMaterials);

    fscanf(f,"%s",line); // skip word: #Textures
    fscanf(f,"%d",&numTextures);

    for (i=1;i<=numTextures;i++)
    {
        fscanf(f,"%s",textureNames[i]);
    }
    
    fscanf(f,"%s",line); // skip word: #Translation
    fscanf(f,"%d",&numTranslations);

    for (i=1;i<=numTranslations;i++)
    {
        fscanf(f,"%lf %lf %lf",&(translations[i].tx),&(translations[i].ty),&(translations[i].tz));
    }
    
    fscanf(f,"%s",line); // skip word: #Scaling
    fscanf(f,"%d",&numScalings);
    
    for (i=1;i<=numScalings;i++)
    {
        fscanf(f,"%lf %lf %lf",&(scalings[i].sx),&(scalings[i].sy),&(scalings[i].sz));
    }
    
    fscanf(f,"%s",line); // skip word: #Rotation
    fscanf(f,"%d",&numRotations);
    
    for (i=1;i<=numRotations;i++)
    {
        fscanf(f,"%lf %lf %lf %lf",&(rotations[i].angle),&(rotations[i].ux),&(rotations[i].uy),&(rotations[i].uz));
    }
    
    fscanf(f,"%d",&numModels);
    for (i=0;i<numModels;i++)
    {
        fscanf(f,"%s",line);
        if (strcmp(line,"#CubeInstance")==0)
        {
            models[i].modelType = 1;
            
        }
        else if (strcmp(line,"#SphereInstance")==0)
        {
            models[i].modelType = 0;
        }
        
        fscanf(f,"%d",&(models[i].mat));
        fscanf(f,"%d",&(models[i].texture));
        if (models[i].modelType==0)
        {
            models[i].center.x = models[i].center.y = models[i].center.z = 0.0;
            models[i].radius = 1.0;
        }
        else if (models[i].modelType==1)
        {
            models[i].vertices[0].x = 1;
            models[i].vertices[0].y = 1;
            models[i].vertices[0].z = 1;

            models[i].vertices[1].x = 1;
            models[i].vertices[1].y = 1;
            models[i].vertices[1].z = -1;

            models[i].vertices[2].x = -1;
            models[i].vertices[2].y = 1;
            models[i].vertices[2].z = -1;

            models[i].vertices[3].x = -1;
            models[i].vertices[3].y = 1;
            models[i].vertices[3].z = 1;

            models[i].vertices[4].x = 1;
            models[i].vertices[4].y = -1;
            models[i].vertices[4].z = 1;
            
            models[i].vertices[5].x = 1;
            models[i].vertices[5].y = -1;
            models[i].vertices[5].z = -1;
            
            models[i].vertices[6].x = -1;
            models[i].vertices[6].y = -1;
            models[i].vertices[6].z = -1;
            
            models[i].vertices[7].x = -1;
            models[i].vertices[7].y = -1;
            models[i].vertices[7].z = 1;
        }
        
        fscanf(f,"%d",&(models[i].numTransformations));
        
        for (j=0;j<models[i].numTransformations;j++)
        {
            fscanf(f,"%s",tmp);
            models[i].transformationTypes[j]=tmp[0];
            fscanf(f,"%d",&(models[i].transformationIDs[j]));

            if (models[i].modelType==0) {
                switch (models[i].transformationTypes[j])
                {
                    case 't' : models[i].center.x+= translations[models[i].transformationIDs[j]].tx;
                        models[i].center.y+= translations[models[i].transformationIDs[j]].ty;
                        models[i].center.z+= translations[models[i].transformationIDs[j]].tz;
                        break;
                    case 's' : models[i].radius *= scalings[models[i].transformationIDs[j]].sx;
                            models[i].center.x *= scalings[models[i].transformationIDs[j]].sx;
                            models[i].center.y *= scalings[models[i].transformationIDs[j]].sz;
                            models[i].center.z *= scalings[models[i].transformationIDs[j]].sz;
                            break;
                }
            } else
            {
                switch (models[i].transformationTypes[j])
                {
                    case 't' :
                        for (k=0;k<8;k++) {
                        models[i].vertices[k].x+= translations[models[i].transformationIDs[j]].tx;
                        models[i].vertices[k].y+= translations[models[i].transformationIDs[j]].ty;
                        models[i].vertices[k].z+= translations[models[i].transformationIDs[j]].tz;
                        }
                        break;
                    case 's' :
                        for (k=0;k<8;k++) {
                        models[i].vertices[k].x *= scalings[models[i].transformationIDs[j]].sx;
                        models[i].vertices[k].y *= scalings[models[i].transformationIDs[j]].sy;
                        models[i].vertices[k].z *= scalings[models[i].transformationIDs[j]].sz;
                        }
                        break;
                }
                
            }
        }
    
    }
    
    fclose(f);
    
}

void initImage()
{
	int i,j;

    for (i=0;i<sizeX;i++)
		for (j=0;j<sizeY;j++)
		{
			image[i][j].r = bgColor.r;
			image[i][j].g = bgColor.g;
			image[i][j].b = bgColor.b;
		}
}

Ray generateRay(int i, int j)
{
	Ray tmp;
	
	Vec3 su,sv,s;
	
	tmp.a = cam.pos;
	
	su = mult(cam.u,cam.l+(i*pixelW)+halfPixelW);
	sv = mult(cam.v,cam.b+(j*pixelH)+halfPixelH);
	
	s = add(su,sv);
	
	tmp.b = add(mult(cam.gaze,cam.d),s);
    
	//printVec(add(tmp.a,tmp.b));
	
	return tmp;
}

    
double intersectSphere(Ray r, Model model, Vec3* normal, Vec3* textCoord)
{
	double A,B,C; //constants for the quadratic function
	
	double delta;
	
	Vec3 scenter;
    double sradius;
    
    Vec3 p;
		
	double t,t1,t2;
    int i;
	
    scenter = model.center;
    sradius = model.radius;
    
    
	C = (r.a.x-scenter.x)*(r.a.x-scenter.x)+(r.a.y-scenter.y)*(r.a.y-scenter.y)+(r.a.z-scenter.z)*(r.a.z-scenter.z)-sradius*sradius;

	B = 2*r.b.x*(r.a.x-scenter.x)+2*r.b.y*(r.a.y-scenter.y)+2*r.b.z*(r.a.z-scenter.z);
	
	A = r.b.x*r.b.x+r.b.y*r.b.y+r.b.z*r.b.z;
	
	delta = B*B-4*A*C;
	
	if (delta<0) return -1;
	else if (delta==0)
	{
		t = -B / (2*A);
	}
	else
	{
		delta = sqrt(delta);
		A = 2*A;
		t1 = (-B + delta) / A;
		t2 = (-B - delta) / A;
		
		if (t1<0 && t2>0) return -1;
		if (t2>0 && t2<0) return -1;
		
		if (t1<t2) t=t1; else t=t2;
	}
	
    if (t>0.000001)
    {
        p = add(r.a,mult(r.b,t));
        *normal = add(p,mult(scenter,-1));
        (*textCoord).x = (atan2(((*normal).z),((*normal).x))+(M_PI))/(2*M_PI);
        (*textCoord).y = acos(((*normal).y)/(sradius))/(M_PI);
    }
	return t;
}

double intersectTriangle(Ray r, Vec3 ma, Vec3 mb, Vec3 mc, double *rgamma, double *rbeta);

double intersectCube(Ray r, Model model, Vec3* normal, Vec3* textCoord)
{
    // generate 12 triangles on the faces and intersect the ray with them one by one and pick the closest intersection
    
    double minT = 10e100;
    double t;
    int i;
    double gamma, beta;
    double mingamma, minbeta;
    Vec3 ma,mb,mc;
    
    int triangles[36] = {0,1,2,2,3,0,0,3,4,4,3,7,0,4,1,4,5,1,3,2,7,7,2,6,2,1,5,5,6,2,4,6,5,4,7,6};
    t = -1;
    
    for (i=0;i<12;i++)
    {
        ma = model.vertices[triangles[i*3]];
        mb = model.vertices[triangles[i*3+1]];
        mc = model.vertices[triangles[i*3+2]];

        t = intersectTriangle(r,ma,mb,mc,&gamma,&beta);
    
        if (t>0.0000001 && t<minT)
        {
            mingamma = gamma;
            minbeta = beta;
            minT = t;
            (*normal) = cross(add(mb,mult(ma,-1)),add(mc,mult(ma,-1)));
            (*normal) = normalize(*normal);
        }
    }
    
    (*textCoord).x = 0.0;
    (*textCoord).y = 0.0;
    
    if (minT<10000) return minT;
    else return -1;
}

double intersectTriangle(Ray r, Vec3 ma, Vec3 mb, Vec3 mc, double *rgamma, double *rbeta)
{
	double  a,b,c,d,e,f,g,h,i,j,k,l;
	double beta,gamma,t;
	
	double eimhf,gfmdi,dhmeg,akmjb,jcmal,blmkc;

	double M;
	
	double dd;
	
	a = ma.x-mb.x;
	b = ma.y-mb.y;
	c = ma.z-mb.z;

	d = ma.x-mc.x;
	e = ma.y-mc.y;
	f = ma.z-mc.z;
	
	g = r.b.x;
	h = r.b.y;
	i = r.b.z;
	
	j = ma.x-r.a.x;
	k = ma.y-r.a.y;
	l = ma.z-r.a.z;
	
	eimhf = e*i-h*f;
	gfmdi = g*f-d*i;
	dhmeg = d*h-e*g;
	akmjb = a*k-j*b;
	jcmal = j*c-a*l;
	blmkc = b*l-k*c;

	M = a*eimhf+b*gfmdi+c*dhmeg;
    if (M==0) return -1;
	
	t = -(f*akmjb+e*jcmal+d*blmkc)/M;
	
	if (t<0) return -1;
	
	gamma = (i*akmjb+h*jcmal+g*blmkc)/M;
	
	if (gamma<0 || gamma>1) return -1;
	
	beta = (j*eimhf+k*gfmdi+l*dhmeg)/M;
	
	if (beta<0 || beta>(1-gamma)) return -1;
	
    *rgamma = gamma;
    *rbeta = beta;
    //printf("%lf %lf %lf\n",gamma,beta,t);
	return t;
}

double intersectModel (Ray r, Model model, Vec3* normal, Vec3* textCoord)
{
            if (model.modelType==0)
            {
                return intersectSphere(r,model,normal,textCoord);
            }
            else
            {
                return intersectCube(r,model,normal,textCoord);
            }
}

int intersectShadow(Ray r, double maxT)
{
/* this function returns whether a given ray intersects another object between t>0 and t<maxT  */
/* maxT is the t where the light direction vector is at the point light source */

	double t;
	int i;
    Vec3 normal,textCoord;
	
	for (i=0;i<numModels;i++)
	{
        t = intersectModel(r,models[i], &normal, &textCoord);
        if (t>0.001 && t<=maxT) return 1;
	}

	return 0;

}

Ray intersect(Ray r, Model* m)
{
/* this function returns the intersection point as a ray */
/* intersection point is Ray.a and the normal of the intersection is Ray.b */
/* texture coordinates are returned in Ray.u and Ray.v */

	double t, mint;
	Ray intersection;
    Vec3 normal;
    Vec3 textCoord;
	int i;
	
	intersection.b.x = 0;
	intersection.b.y = 0;
	intersection.b.z = 0;
	/* normal length 0 if there's no intersection */
	
	mint = 10e100;

	for (i=0;i<numModels;i++)
	{
		t = intersectModel(r,models[i], &normal, &textCoord);
		if (t<mint && t>0.000001)
		{
			mint = t;
			*m = models[i];
			intersection.a = add(r.a,mult(r.b,t));
			intersection.b = normal;
            
            if (m->texture!=0)
            {
                intersection.u = textCoord.x;
                intersection.v = textCoord.y;
            }
            
            intersection.b = normalize(intersection.b);
		}
	}
	
	return intersection;

}

Color computeColor(Ray,Model,Vec3,int);

Color rayColor(Ray r, int count)
{
	Color c;
	Ray intersection;
	Model m;

	c.r = 0.0;
	c.g = 0.0;
	c.b = 0.0;

	if (count>reflectCount) return c;
	else
	{
		intersection = intersect(r,&m);
				
		if (!(length(intersection.b)<0.000001))
		{
	            c = computeColor(intersection,m,mult(normalize(r.b),-1),count);
		}
		return c;
	}
}

Color computeColor(Ray r, Model m, Vec3 v,int count)
{
	// Ray r stores the intersection point in r.a and the normal of the surface at r.b
	// r.u and r.v stores texture coordinates
	Color c;
	Color rc;
	int i;
	double d;
	double maxT;
	Vec3 lightDir;
	Vec3 h;
	Ray lightRay;
	Ray reflectRay;
    Material mat;
	
    mat = materials[m.mat];

    c.r = ambColor.r*mat.ambient.r;
	c.g = ambColor.g*mat.ambient.g;
	c.b = ambColor.b*mat.ambient.b;
	
	
	// first find contributions from the light sources
	
	for (i=0;i<numLights;i++)
	{
		// find weather the light ray is blocked by another object, i.e. there is shadow
		lightDir = add(lights[i].pos,mult(r.a,-1));
		maxT = length(lightDir);
		lightDir = normalize(lightDir);
		lightRay.a = r.a;
		lightRay.b = lightDir;

		if (intersectShadow(lightRay,maxT))
		{
			continue; // in shadow, continue with the next light
		}
		
		// the code below is executed when the object is not in shadow
		// add diffuse component for each point light source
        
		d = dot(r.b,lightDir);
		if (d>0)
		{
			if (m.texture==0) {
				c.r+=mat.diffuse.r*lights[i].c.r*d;
				c.g+=mat.diffuse.g*lights[i].c.g*d;
				c.b+=mat.diffuse.b*lights[i].c.b*d;
			}
			else
			{
				c.r+=texture[m.texture][(int)(r.v*textHeight[m.texture])][(int)((1-r.u)*textWidth[m.texture])].r*lights[i].c.r*d;
				c.g+=texture[m.texture][(int)(r.v*textHeight[m.texture])][(int)((1-r.u)*textWidth[m.texture])].g*lights[i].c.g*d;
				c.b+=texture[m.texture][(int)(r.v*textHeight[m.texture])][(int)((1-r.u)*textWidth[m.texture])].b*lights[i].c.b*d;
			}
		}
		
		// add specular component for each point light source
		h = normalize(add(lightDir,v));
		d = dot(r.b,h);
		if (d>0)
		{
			c.r+=mat.specular.r*lights[i].c.r*pow(d,mat.specExp);
			c.g+=mat.specular.g*lights[i].c.g*pow(d,mat.specExp);
			c.b+=mat.specular.b*lights[i].c.b*pow(d,mat.specExp);
		}
	}

	// now add reflections from other objects

	if (mat.reflect.r>0.0 || mat.reflect.g>0.0 || mat.reflect.b>0.0)
	{
		reflectRay.a = r.a;
		reflectRay.b = normalize(add(mult(v,-1),mult(r.b,2*dot(v,r.b))));
		rc = rayColor(reflectRay,count+1);
		c.r+=rc.r*mat.reflect.r;
		c.g+=rc.g*mat.reflect.g;
		c.b+=rc.b*mat.reflect.b;
	}

	
	return c;
}
void rayTrace()
{
		Ray r;
		Ray intersection;
		Color c;
		Model m;
    
		int i,j;
		for (j=0;j<sizeY;j++)
		{
			for (i=0;i<sizeX;i++)
			{
				//printf("[%d  ,  %d]\r\n",i,j);
				r = generateRay(i,j);
				
				intersection = intersect(r,&m);
				
				if (!(length(intersection.b)<0.000001))
				{
					c = computeColor(intersection,m,mult(normalize(r.b),-1),0);
				
					image[i][j].r = c.r;
					image[i][j].g = c.g;
					image[i][j].b = c.b;
				}
			}
		}
}

void writeImage(char *fileName)
{
	FILE *outFile;
	int i,j;
	
	outFile = fopen(fileName,"w");
	
	fprintf(outFile,"P3\n");
	fprintf(outFile,"# %s\n",outFileName);
	
	fprintf(outFile,"%d %d\n",sizeX,sizeY);
	
	fprintf(outFile,"255\n");
	
	for (j=sizeY-1;j>=0;j--)
	{
		for (i=0;i<sizeX;i++)
		{
			fprintf(outFile,"%d %d %d ",convert(image[i][j].r),convert(image[i][j].g),convert(image[i][j].b));
		}
		fprintf(outFile,"\n");
	}
	fclose(outFile);
}

int main(int argc, char **argv)
{
    char comm[100];
    int i,j,k;
    if (argc<2) {
	printf("Usage: raytracing <scene file> <camera file>\n");
	return 1;
    }

    texture = (Color ***)malloc(sizeof(Color**)*100);
    readCamera(argv[2]);
    readScene(argv[1]);
    
    for (i=1;i<=numTextures;i++)
    {
        readTexture(textureNames[i],i);
    }
    
    printf("Num Models = %d\n",numModels);
    printf("Num Lights = %d\n",numLights);

    if (SDL_Init(SDL_INIT_EVERYTHING) == -1) {
        return 1;
    }

    const int SCREEN_BPP = 32;

    //The surface that will be used
    SDL_Surface *screen = NULL;

    //Set up the screen
    screen = SDL_SetVideoMode(sizeX, sizeY, SCREEN_BPP, SDL_SWSURFACE);

    if (screen == NULL) { return 1;
    }

    SDL_FillRect( SDL_GetVideoSurface(), NULL, 0 );
    SDL_WM_SetCaption( "Ray Tracer", NULL );

    Uint32 color;

	initImage();
	rayTrace();

	for (j=sizeY-1;j>=0;j--)
	{
		for (i=0;i<sizeX;i++)
		{
    			color = SDL_MapRGB(screen->format, convert(image[i][j].r), convert(image[i][j].g), convert(image[i][j].b));
    			putpixel(screen, i, sizeY-j, color);
		}
	}

	if (SDL_Flip(screen) == -1)
	{
		return 1;
	}
    SDL_GetError();
    // SDL_Delay(10000);
    SDL_Event evt;
    int programrunning = 0;
    while(programrunning)
    {
      SDL_WaitEvent(&evt);
      if(evt.type == SDL_QUIT)
        programrunning = 1;
    }
    writeImage("test.ppm");
    //Quit SDL
    SDL_Quit();
    
    return 0;
}
