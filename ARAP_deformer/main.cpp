#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <limits>
#include <Windows.h>
#include <gl/GL.h>
#include <glut.h>

#include "glm.h"
#include "mtxlib.h"
#include "trackball.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include "LeastSquaresSparseSolver.h"

using namespace std;
using namespace Eigen;

// ----------------------------------------------------------------------------------------------------
// global variables

_GLMmodel *mesh, *mesh_old;

int WindWidth, WindHeight;
int last_x, last_y;
int select_x, select_y;

typedef enum { SELECT_MODE, DEFORM_MODE } ControlMode;
ControlMode current_mode = SELECT_MODE;

vector<float*> colors;
vector<vector<int> > handles;
int selected_handle_id = -1;
bool deform_mesh_flag = false;

vector<vector <int>> connectPoints;	//連結點
bool check = false;	//檢查選擇點是否重複、連結(false為無重複、無連結；true為重複、有連結)
vector<vector<vector <double>>> e_old;	//儲存頂點的位置資訊，row, col, xyz
vector<vector<vector <double>>> e_new;
vector<vector <double>> result;	//承接Cholosky計算結果
bool repeat = false;	//迭代是否停止，true繼續、false停止
float distance_lim = 0.00005;	//迭代用數值
int selectSize = 0;

// ----------------------------------------------------------------------------------------------------
// render related functions

void Reshape(int width, int height)
{
	int base = min(width, height);

	tbReshape(width, height);
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLdouble)width / (GLdouble)height, 1.0, 128.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.5);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	tbMatrix();

	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glmDraw(mesh, GLM_SMOOTH);

	// render wire model
	glPolygonOffset(1.0, 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3f(0.6, 0.0, 0.8);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glmDraw(mesh, GLM_SMOOTH);

	// render handle points
	glPointSize(10.0);
	glEnable(GL_POINT_SMOOTH);
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for (int handleIter = 0; handleIter < handles.size(); handleIter++)
	{
		glColor3fv(colors[handleIter%colors.size()]);
		for (int vertIter = 0; vertIter < handles[handleIter].size(); vertIter++)
		{
			int idx = handles[handleIter][vertIter];
			glVertex3fv((float *)&mesh->vertices[3 * idx]);
		}
	}
	glEnd();

	glPopMatrix();

	glFlush();
	glutSwapBuffers();
}

// ----------------------------------------------------------------------------------------------------
// mouse related functions

vector3 Unprojection(vector2 _2Dpos)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x, viewport[3] - (int)_2Dpos.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = { 0.0, 0.0, 0.0 };

	gluUnProject(X, ((double)viewport[3] - Y), (double)Depth, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return vector3(wpos[0], wpos[1], wpos[2]);
}

vector2 projection_helper(vector3 _3Dpos)
{
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	double wpos[3] = { 0.0, 0.0, 0.0 };
	gluProject(_3Dpos.x, _3Dpos.y, _3Dpos.z, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return vector2(wpos[0], (double)viewport[3] - wpos[1]);
}

// Rotation Matrix
Matrix3f RotationMatrix(int count) {

	Matrix3f Si = Matrix3f::Zero();
	vector3 ei_old, ei_new;

	for (int a = 0; a < connectPoints[count].size(); a++) {
		// x, y, z
		ei_old = vector3(e_old[count][a][0], e_old[count][a][1], e_old[count][a][2]);
		ei_new = vector3(e_new[count][a][0], e_new[count][a][1], e_new[count][a][2]);

		for (int b = 0; b < 3; b++)
		for (int c = 0; c < 3; c++)
			Si(b, c) += ei_old[b] * ei_new[c];
	}

	// compute SVD decomposition of a matrix m
	// SVD: m = U * S * V^T
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(Si, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const Eigen::Matrix3f U = svd.matrixU();
	// note that this is actually V^T!!
	const Eigen::Matrix3f V = svd.matrixV();
	const Eigen::VectorXf S = svd.singularValues();

	return V*(U.transpose());
}

void mouse(int button, int state, int x, int y)
{
	tbMouse(button, state, x, y);

	if (current_mode == SELECT_MODE && button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			select_x = x;
			select_y = y;
		}
		else
		{
			vector<int> this_handle;

			// project all mesh vertices to current viewport
			for (int vertIter = 0; vertIter < mesh->numvertices; vertIter++)
			{
				vector3 pt(mesh->vertices[3 * vertIter + 0], mesh->vertices[3 * vertIter + 1], mesh->vertices[3 * vertIter + 2]);
				vector2 pos = projection_helper(pt);

				// if the projection is inside the box specified by mouse click&drag, add it to current handle
				if (pos.x >= select_x && pos.y >= select_y && pos.x <= x && pos.y <= y)
				{
					this_handle.push_back(vertIter);

					//計數
					selectSize += 1;
				}
			}
			handles.push_back(this_handle);
		}
	}
	// select handle
	else if (current_mode == DEFORM_MODE && button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{

		// project all handle vertices to current viewport
		// see which is closest to selection point
		double min_dist = 999999;
		int handle_id = -1;
		for (int handleIter = 0; handleIter < handles.size(); handleIter++)
		{
			for (int vertIter = 0; vertIter < handles[handleIter].size(); vertIter++)
			{
				int idx = handles[handleIter][vertIter];
				vector3 pt(mesh->vertices[3 * idx + 0], mesh->vertices[3 * idx + 1], mesh->vertices[3 * idx + 2]);
				vector2 pos = projection_helper(pt);

				double this_dist = sqrt((double)(pos.x - x)*(pos.x - x) + (double)(pos.y - y)*(pos.y - y));
				if (this_dist < min_dist)
				{
					min_dist = this_dist;
					handle_id = handleIter;
				}
			}
		}

		selected_handle_id = handle_id;
		deform_mesh_flag = true;
	}
	else if (current_mode == DEFORM_MODE && button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
	{
		float distance_min = 1;
		// 收斂位置
		repeat = true;
		do
		{
			int tempI = 0;
			float tempF = 0;

			// 更新e
			float distance = 1;	//停止迭代用數值
			for (int i = 0; i < mesh->numvertices; i++)
			for (int j = 0; j < connectPoints[i].size(); j++)
			{
				for (int k = 0; k < 3; k++)
				{
					e_new[i][j].push_back(mesh->vertices[3 * (i + 1) + k] - mesh->vertices[3 * (connectPoints[i][j] + 1) + k]);
					tempF += abs(e_new[i][j][k]);
				}
				if (tempF < distance)
					distance = tempF;
			}

			// 是否停止迭代
			if (abs(distance_min - distance)>distance_lim)
				distance_min = distance;
			else
				repeat = false;

			// 解Matrix
			//vector<Triplet<double>> solver;	//The linear system
			LeastSquaresSparseSolver solver;
			solver.Create((mesh->numvertices + selectSize), mesh->numvertices, 3);

			// L Matrix (1, -1/n, 0)
			// pre-process *n
			for (int i = 0; i < mesh->numvertices; i++)
			{
				//自己為1
				solver.AddSysElement(i, i, connectPoints[i].size());
				//solver.push_back(Triplet<double>(i, i, connectPoints[i].size()));
				//solver.AddSysElement(i, i, 1);

				//連結點為(-1/n)
				for (int j = 0; j < connectPoints[i].size(); j++)
					solver.AddSysElement(i, connectPoints[i][j], -1);
				//solver.push_back(Triplet<double>(i, connectPoints[i][j], -1));
				//solver.AddSysElement(i, connectPoints[i][j], -pow(connectPoints[i].size(), -1));
			}

			//位置限制
			tempI = 0;
			for (int i = 0; i < handles.size(); i++)
			for (int j = 0; j < handles[i].size(); j++)
			{
				//solver.push_back(Triplet<double>(tempI + mesh->numvertices, handles[i][j] - 1, 1));
				solver.AddSysElement(tempI + mesh->numvertices, handles[i][j] - 1, 1);
				tempI += 1;
			}

			// 1/2 * ( Ri + Rj ) * ( pi - pj )
			// e_old[i][j] : pi-pj
			float **B = new float*[3];
			for (int i = 0; i < 3; i++)
				B[i] = new float[mesh->numvertices + selectSize];
			//vector<Triplet<double>> B;

			Matrix3Xf eij = Matrix3Xf::Zero(3, 1);
			Matrix3Xf tempM = Matrix3Xf::Zero(3, 1);

			for (int a = 0; a < mesh->numvertices; a++)
			{
				tempM = Matrix3Xf::Zero(3, 1);
				for (int b = 0; b < connectPoints[a].size(); b++)
				{
					for (int c = 0; c < 3; c++)
						eij(c, 0) = e_old[a][b][c];
					// Rotation Matrix
					tempM += (RotationMatrix(a) + RotationMatrix(connectPoints[a][b]))*eij;
				}
				tempM *= 0.5;

				for (int b = 0; b < 3; b++)
					B[b][a] = tempM(b, 0);
				//B.push_back(Triplet<double>(a, b, tempM(b, 0)));
			}

			// 控制點位置
			tempI = 0;
			for (int a = 0; a < handles.size(); a++)
			for (int b = 0; b < handles[a].size(); b++)
			{
				for (int c = 0; c < 3; c++)
					B[c][tempI + mesh->numvertices] = mesh->vertices[3 * handles[a][b] + c];
				//B.push_back(Triplet<double>(c, tempI + mesh->numvertices, mesh->vertices[3 * (handles[a][b] + c)]));
				tempI++;
			}

			// 數值處理
			// Solver
			// modify : float *AX = (float*)A->x; *4
			solver.SetRightHandSideMatrix(B);
			solver.CholoskyFactorization();
			for (int i = 0; i < 3; i++)
				solver.CholoskySolve(i);
			for (int i = 0; i < mesh->numvertices; i++)
			for (int j = 0; j < 3; j++)
				result[i][j] = solver.GetSolution(j, i);
			solver.ResetSolver(0, 0, 0);

			// B
			for (int i = 0; i < 3; i++)
				delete[] B[i];
			delete[] B;

			/*
			SparseMatrix<double> A(mesh->numvertices, mesh->numvertices);
			A.setFromTriplets(solver.begin(), solver.end());

			//Convert into ATA=ATb
			//SparseMatrix<double> ATA = A.transpose()*A;
			//C = A.transpose()*C;

			SparseMatrix<double> Bx(mesh->numvertices, 3);
			Bx.setFromTriplets(B.begin(), B.end());
			Bx = A.transpose()*Bx;

			SimplicialCholesky<SparseMatrix<double>> chol(A.transpose()*A);
			MatrixXd result = chol.solve(Bx);

			vector<Triplet<double>> solver;	//The linear system
			vector<Triplet<double>> B;
			for (int i = 0; i < mesh->numvertices; i++)
			{
			//自己為1
			solver.push_back(Triplet<double>(i, i, 1));

			//連結點為(-1/n)
			for (int j = 0; j < connectPoints[i].size(); j++)
			solver.push_back(Triplet<double>(i, connectPoints[i][j], -pow(connectPoints[i].size(), -1)));
			}

			VectorXd C((mesh->numvertices), 3);
			for (int a = 0; a < mesh->numvertices; a++)
			for (int b = 0; b < 3; b++)
			C(a, b) = 0;

			for (int a = 0; a < connectPoints.size(); a++)
			for (int b = 0; b < 3; b++) {
			C((mesh->numvertices + a), b) = mesh->vertices[(connectPoints[a][b] + 1) * 3 + b];
			B.push_back(Triplet<double>((mesh->numvertices + a), b, mesh->vertices[(connectPoints[a][b] + 1) * 3 + b]));
			}

			SparseMatrix<double> A(mesh->numvertices + connectPoints.size(), mesh->numvertices);
			A.setFromTriplets(solver.begin(), solver.end());
			SparseMatrix<double> Bx(mesh->numvertices + connectPoints.size(), 3);
			Bx.setFromTriplets(B.begin(), B.end());
			Bx = A.transpose()*Bx;

			SimplicialCholesky<SparseMatrix<double>> chol(A.transpose()*A);
			MatrixXd result = chol.solve(Bx);
			*/

			// 重新建模
			for (int i = 0; i < mesh->numvertices; i++)
			for (int j = 0; j < 3; j++)
				mesh->vertices[3 * (i + 1) + j] = result[i][j];
			//mesh->vertices[3 * (i + 1) + j] = result(i, j);

			Display();
		} while (repeat != false);
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
		deform_mesh_flag = false;

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{
	tbMotion(x, y);

	// if in deform mode and a handle is selected, deform the mesh
	if (current_mode == DEFORM_MODE && deform_mesh_flag == true)
	{
		matrix44 m;
		vector4 vec = vector4((float)(x - last_x) / 1000.0f, (float)(y - last_y) / 1000.0f, 0.0, 1.0);

		gettbMatrix((float *)&m);
		vec = m * vec;

		// deform handle points
		for (int vertIter = 0; vertIter < handles[selected_handle_id].size(); vertIter++)
		{
			int idx = handles[selected_handle_id][vertIter];
			// Y方向
			vector3 pt(mesh->vertices[3 * idx + 0] + vec.x, mesh->vertices[3 * idx + 1] - vec.y, mesh->vertices[3 * idx + 2] - vec.z);
			mesh->vertices[3 * idx + 0] = pt[0];
			mesh->vertices[3 * idx + 1] = pt[1];
			mesh->vertices[3 * idx + 2] = pt[2];
		}
	}

	last_x = x;
	last_y = y;
}

// ----------------------------------------------------------------------------------------------------
// keyboard related functions

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'd':
	{
		printf("Current Mode : Deform\n");
		current_mode = DEFORM_MODE;
		break;
	}
	default:
	case 's':
	{
		printf("Current Mode : Select\n");
		current_mode = SELECT_MODE;
		break;
	}
	}
}

// ----------------------------------------------------------------------------------------------------
// main function

void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}

int main(int argc, char *argv[])
{
	WindWidth = 800;
	WindHeight = 800;

	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_position[] = { 0.0, 0.0, 1.0, 0.0 };

	// color list for rendering handles
	float red[] = { 1.0, 0.0, 0.0 };
	colors.push_back(red);
	float yellow[] = { 1.0, 1.0, 0.0 };
	colors.push_back(yellow);
	float blue[] = { 0.0, 1.0, 1.0 };
	colors.push_back(blue);
	float green[] = { 0.0, 1.0, 0.0 };
	colors.push_back(green);

	glutInit(&argc, argv);
	glutInitWindowSize(WindWidth, WindHeight);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutCreateWindow("ARAP");

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);
	glClearColor(0, 0, 0, 0);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	tbInit(GLUT_LEFT_BUTTON);
	tbAnimate(GL_TRUE);

	glutTimerFunc(40, timf, 0); // Set up timer for 40ms, about 25 fps

	// load 3D model
	mesh = glmReadOBJ("../data/man.obj");
	//mesh = glmReadOBJ("../data/Dino.obj");
	mesh_old = glmReadOBJ("../data/man.obj");
	//mesh_old = glmReadOBJ("../data/Dino.obj");

	glmUnitize(mesh);
	glmFacetNormals(mesh);
	glmVertexNormals(mesh, 90.0);

	glmUnitize(mesh_old);
	glmFacetNormals(mesh_old);
	glmVertexNormals(mesh_old, 90.0);

	// 紀錄連結點資訊
	connectPoints.resize(mesh->numvertices);
	int indexTriangle = 0;
	int indexConnect = 0;
	bool check = false;	// 檢查重複，重複為true，不重複為false
	for (int a = 0; a < mesh->numtriangles; a++)	// 總共有a個三角形
	{
		for (int b = 0; b < 3; b++)	// 第a個三角形的3個頂點
		{
			indexTriangle = mesh->triangles[(a)].vindices[b] - 1;	// 第a個三角形 第b個頂點

			for (int c = 0; c < 3; c++)
			{
				indexConnect = mesh->triangles[(a)].vindices[c] - 1;	// 第a個三角形 第c個頂點
				// 先鎖定第a個點，檢查另外2個點

				// 檢查是否已經列入connectPoints
				for (int d = 0; d < connectPoints[indexTriangle].size(); d++)
				if (connectPoints[indexTriangle][d] == indexConnect)
					check = true;

				// 連結點列表排除自己
				if (b != c && check == false)
					connectPoints[indexTriangle].push_back(indexConnect);
			}
			check = false;
		}
	}

	// 初始化陣列大小
	e_old.resize(mesh->numvertices);
	e_new.resize(mesh->numvertices);
	for (int i = 0; i < mesh->numvertices; i++)
	{
		e_old[i].resize(connectPoints[i].size());
		e_new[i].resize(connectPoints[i].size());
	}
	for (int i = 0; i < mesh->numvertices; i++)	// row
	for (int j = 0; j < connectPoints[i].size(); j++)	// col
	for (int k = 0; k < 3; k++)	// x, y, z
	{
		//e_old[i][j].push_back(mesh->vertices[3 * (i + 1) + k] - mesh->vertices[3 * (connectPoints[i][j] + 1) + k]);
		e_old[i][j].push_back(mesh_old->vertices[3 * (i + 1) + k] - mesh_old->vertices[3 * (connectPoints[i][j] + 1) + k]);
		e_new[i][j].push_back(mesh->vertices[3 * (i + 1) + k] - mesh->vertices[3 * (connectPoints[i][j] + 1) + k]);
	}

	// 初始化
	result.resize(mesh->numvertices, vector<double>(3, 0));

	glutMainLoop();	//循環點

	return 0;

}