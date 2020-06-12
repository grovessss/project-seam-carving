#include<queue>
// 以下部分直接复制，替代project1文件代码中的 findseam中对应部分即可 
// 动态规划dp 
	if (seam_direction == VERTICAL) {
		path.resize(rowsize);
		double distto[2000][2000];      //记录最短距离 
		int edgeto[2000][2000];      //记录最短路
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				if (i == 0) distto[i][j] = cumulative_energy_map.at<double>(i, j);
				else distto[i][j] = 0;
				edgeto[i][j] = 0;
			}
		}
		for (int r = 1; r < rowsize; r++) {
			for (int c = 0; c < colsize; c++) {
				distto[r][c] = distto[r - 1][c] + cumulative_energy_map.at<double>(r, c);
				edgeto[r][c] = 0;
				if (c != 0) {
					if (distto[r - 1][c-1] + cumulative_energy_map.at<double>(r, c) < distto[r][c]) {
						distto[r][c] = distto[r - 1][c-1] + cumulative_energy_map.at<double>(r, c);
						edgeto[r][c] = -1;
					}
				}
				if (c != colsize - 1) {
					if (distto[r - 1][c+1] + cumulative_energy_map.at<double>(r, c) < distto[r][c]) {
						distto[r][c] = distto[r - 1][c+1] + cumulative_energy_map.at<double>(r, c);
						edgeto[r][c] = 1;
					}
				}
			}
		}
		int min_index = 0;
		min_val = distto[rowsize - 1][0];
		for (int i = 1; i < colsize; i++) {
			if (min_val > distto[rowsize - 1][i]) {
				min_index = i;
				min_val = distto[rowsize - 1][i];
			}
		}
		//path.resize(rowsize);
		path[rowsize - 1] = min_index;
		for (int i = rowsize - 1; i > 0; i--) {
			path[i - 1] = path[i] + edgeto[i][path[i]];
		}
    }
  	else if (seam_direction == HORIZONTAL) {
		path.resize(colsize);
		double distto[2000][2000];      //记录最短距离 
		int edgeto[2000][2000];      //记录最短路
		for (int i = 0; i < colsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				if (i == 0) distto[j][i] = cumulative_energy_map.at<double>(j, i);
				else distto[j][i] = 0;
				edgeto[j][i] = 0;
			}
		}
		for (int c = 1; c < colsize; c++) {
			for (int r = 0; r < rowsize; r++) {
				distto[r][c] = distto[r][c - 1] + cumulative_energy_map.at<double>(r, c);
				edgeto[r][c] = 0;
				if (r != 0) {
					if (distto[r - 1][c - 1] + cumulative_energy_map.at<double>(r, c) < distto[r][c]) {
						distto[r][c] = distto[r - 1][c - 1] + cumulative_energy_map.at<double>(r, c);
						edgeto[r][c] = -1;
					}
				}
				if (r != rowsize - 1) {
					if (distto[r + 1][c - 1] + cumulative_energy_map.at<double>(r, c) < distto[r][c]) {
						distto[r][c] = distto[r + 1][c - 1] + cumulative_energy_map.at<double>(r, c);
						edgeto[r][c] = 1;
					}
				}
			}
		}
		int min_index = 0;
		min_val = distto[0][colsize - 1];
		for (int i = 1; i < rowsize; i++) {
			if (distto[i][colsize - 1] < min_val) {
				min_val = distto[i][colsize - 1];
				min_index = i;
			}
		}
		path[colsize - 1] = min_index;
		for (int i = colsize - 1; i >0 ; i--) {
			path[i - 1] = path[i] + edgeto[path[i]][i];
		}
	}
    
  //原本代码 
  if (seam_direction == VERTICAL) {
        // copy the data from the last row of the cumulative energy map
        Mat row = cumulative_energy_map.row(rowsize - 1);
    
        // get min and max values and locations
        minMaxLoc(row, &min_val, &max_val, &min_pt, &max_pt);
        
        // initialize the path vector
        path.resize(rowsize);
        int min_index = min_pt.x;
        path[rowsize - 1] = min_index;
        
        // starting from the bottom, look at the three adjacent pixels above current pixel, choose the minimum of those and add to the path
        for (int i = rowsize - 2; i >= 0; i--) {
            a = cumulative_energy_map.at<double>(i, max(min_index - 1, 0));
            b = cumulative_energy_map.at<double>(i, min_index);
            c = cumulative_energy_map.at<double>(i, min(min_index + 1, colsize - 1));
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), colsize - 1); // take care of edge cases
            path[i] = min_index;
        }
    }
    else if (seam_direction == HORIZONTAL) {
        // copy the data from the last column of the cumulative energy map
        Mat col = cumulative_energy_map.col(colsize - 1);
        
        // get min and max values and locations
        minMaxLoc(col, &min_val, &max_val, &min_pt, &max_pt);
        
        // initialize the path vector
        path.resize(colsize);
        int min_index = min_pt.y;
        path[colsize - 1] = min_index;
        
        // starting from the right, look at the three adjacent pixels to the left of current pixel, choose the minimum of those and add to the path
        for (int i = colsize - 2; i >= 0; i--) {
            a = cumulative_energy_map.at<double>(max(min_index - 1, 0), i);
            b = cumulative_energy_map.at<double>(min_index, i);
            c = cumulative_energy_map.at<double>(min(min_index + 1, rowsize - 1), i);
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), rowsize - 1); // take care of edge cases
            path[i] = min_index;
        }
    }
    
    
    //dijskra
   struct Ppoint {
		public:
			int x, y;
			double d;
			Ppoint(int a, int b, double c) {
				x = a; y = b; d = c;
			}
			Ppoint() {}
			bool operator<(const Ppoint& sec) const{
				if (d != sec.d) return(d > sec.d);
				else if (x != sec.x) return(x < sec.x);
				else return(y > sec.y);
			}
	};
	if (seam_direction == VERTICAL) {
		path.resize(rowsize);
		double dist[1000][1000]; //记录距离 
		int edge[1000][1000]; //记录从哪个顶点来 
		bool s[1000][1000];
		priority_queue<Ppoint> attach;
		for (int i = 0; i < colsize; i++) {
			dist[0][i] = cumulative_energy_map.at<double>(0, i);
			Ppoint temp(0, i, dist[0][i]);
			attach.push(temp);
			s[0][i] = false;
		}
		for (int i = 1; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				dist[i][j] = DBL_MAX;
				s[i][j] = false;
				edge[i][j] = 0;
			}
		}
		int ans;
		while (true) {
			//int min_index = 0;
			//min_val = attach.top().first;
			int r = attach.top().x;
			int c = attach.top().y;
			attach.pop();
			if (s[r][c] == true) continue;
			else s[r][c] = true;
			if (r == rowsize-1) {
				ans = c;
				break;
			}

			if (dist[r][c] + cumulative_energy_map.at<double>(r + 1, c) < dist[r + 1][c] && s[r + 1][c] == false) {
				dist[r + 1][c] = dist[r][c] + cumulative_energy_map.at<double>(r + 1, c);
				Ppoint temp(r + 1, c, dist[r + 1][c]);
				attach.push(temp);
				edge[r + 1][c] = c;
			}
			if (c != 0) {
				if (dist[r][c] + cumulative_energy_map.at<double>(r + 1, c - 1) < dist[r + 1][c - 1] && s[r + 1][c - 1] == false) {
					dist[r + 1][c - 1] = dist[r][c] + cumulative_energy_map.at<double>(r + 1, c - 1);
					Ppoint temp(r + 1, c - 1, dist[r + 1][c - 1]);
					attach.push(temp);
					edge[r + 1][c - 1] = c;
				}
			}
			if (c != (colsize - 1)) {
				if (dist[r][c] + cumulative_energy_map.at<double>(r + 1, c + 1) < dist[r + 1][c + 1] && s[r + 1][c + 1] == false) {
					dist[r + 1][c + 1] = dist[r][c] + cumulative_energy_map.at<double>(r + 1, c + 1);
					Ppoint temp(r + 1, c + 1, dist[r + 1][c + 1]);
					attach.push(temp);
					edge[r + 1][c + 1] = c;
				}
			}
		}
		path[rowsize - 1] = ans;
		for (int i = rowsize - 1; i > 0; i--) {
			path[i - 1] = edge[i][path[i]];
		}
	}
	else if (seam_direction == HORIZONTAL) {
		path.resize(colsize);
		double dist[1000][1000];
		int edge[1000][1000];
		bool s[1000][1000];
		priority_queue<Ppoint> attach;
		for (int i = 0; i < rowsize; i++) {
			dist[i][0] = cumulative_energy_map.at<double>(i, 0);
			Ppoint temp(i, 0, dist[i][0]);
			attach.push(temp);
			s[i][0] = false;
		}
		for (int i = 1; i < colsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				dist[j][i] = DBL_MAX;
				s[j][i] = false;
				edge[j][i] = 0;
			}
		}
		int ans;
		while (true) {
			//int min_index = 0;
			//min_val = attach.top().first;
			int r = attach.top().x;
			int c = attach.top().y;
			attach.pop();
			if (s[r][c] == true) continue;
			else s[r][c] = true;
			if (c == colsize - 1) {
				ans = r;
				break;
			}

			if (dist[r][c] + cumulative_energy_map.at<double>(r, c + 1) < dist[r][c + 1] && s[r][c + 1] == false) {
				dist[r][c + 1] = dist[r][c] + cumulative_energy_map.at<double>(r, c + 1);
				Ppoint temp(r, c + 1, dist[r][c + 1]);
				attach.push(temp);
				edge[r][c + 1] = r;
			}
			if (r != 0) {
				if (dist[r][c] + cumulative_energy_map.at<double>(r - 1, c + 1) < dist[r - 1][c + 1] && s[r - 1][c + 1] == false) {
					dist[r - 1][c + 1] = dist[r][c] + cumulative_energy_map.at<double>(r - 1, c + 1);
					Ppoint temp(r - 1, c + 1, dist[r - 1][c + 1]);
					attach.push(temp);
					edge[r - 1][c + 1] = r;
				}
			}
			if (r != (rowsize - 1)) {
				if (dist[r][c] + cumulative_energy_map.at<double>(r + 1, c + 1) < dist[r + 1][c + 1] && s[r + 1][c + 1] == false) {
					dist[r + 1][c + 1] = dist[r][c] + cumulative_energy_map.at<double>(r + 1, c + 1);
					Ppoint temp(r + 1, c + 1, dist[r + 1][c + 1]);
					attach.push(temp);
					edge[r + 1][c + 1] = r;
				}
			}
		}
		path[colsize - 1] = ans;
		for (int i = colsize - 1; i > 0; i--) {
			path[i - 1] = edge[path[i]][i];
		}
	}
	
	//拓扑排序
	struct single_point{
		int x,y;
		single_point(int a,int b){|
			x=a;y=b;
		}
		single_point(){}
	}; 
	if (seam_direction == VERTICAL) {
		path.resize(rowsize);
		double dist[1000][1000]; //记录距离 
		int edge[1000][1000]; //记录从哪个顶点来 
		queue<single_point> q;
		for (int i = 0; i < colsize; i++) {
			dist[0][i] = cumulative_energy_map.at<double>(0, i);
			single_point temp(1,i);
			q.push(temp);
		}
		for (int i = 1; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				dist[i][j] = DBL_MAX;
				edge[i][j] = 0;
			}
		}
		while(!q.empty()){
			single_point temp_point=q.front();
			q.pop();
			int r=temp_point.x;
			int c=temp_point.y;
			double m=dist[r-1][c];
			edge[r][c]=c;
			if(c!=0){
				if(dist[r-1][c-1]<m){
					m=dist[r-1][c-1];
					edge[r][c]=c-1;
				} 
			} 
			if(c!=colsize-1){
				if(dist[r-1][c+1]<m){
					m=dist[r-1][c+1];
					edge[r][c]=c+1;	
				} 
			} 
			dist[r][c]=m+cumulative_energy_map.at<double>(r, c);
			if(r==rowsize-1) continue;
			single_point temp(r+1,c);
			q.push(temp);
		}
		int min_index=0;min_val=DBL_MAX;
		for(int i=0;i<colsize;i++){
			if(dist[rowsize-1][i]<min_val){
				min_index=i;
				min_val=dist[rowsize-1][i];
			}
		}
		path[rowsize-1]=min_index;
		for (int i = rowsize - 1; i > 0; i--) {
			path[i - 1] = edge[i][path[i]];
		}
	} 
	else if (seam_direction == HORIZONTAL) {
		path.resize(colsize);
		double dist[1000][1000];
		int edge[1000][1000];
		queue<single_point> q;
		for(int i=0;i<rowsize;i++){
			dist[i][0]= cumulative_energy_map.at<double>(i,0);
			single_point temp(i,1);
			q.push(temp); 
		}
		for (int i = 1; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				dist[i][j] = DBL_MAX;
				edge[i][j] = 0;
			}
		}
		for(int i=1;i<colsize;i++){
			for(int j=0;j<rowsize;j++){
				dist[j][i]=DBL_MAX;
				edge[j][i]=0;
			}
		}
		while(!q.empty()){
			single_point temp_point=q.front();
			q.pop();
			int r=temp_point.x;
			int c=temp_point.y;
			double m=dist[r][c-1];
			edge[r][c]=r;
			if(r!=0){
				if(dist[r-1][c-1]<m){
					m=dist[r-1][c-1];
					edge[r][c]=r-1;
				} 
			} 
			if(r!=rowsize-1){
				if(dist[r+1][c-1]<m){
					m=dist[r+1][c-1];
					edge[r][c]=r+1;	
				} 
			} 
			dist[r][c]=m+cumulative_energy_map.at<double>(r, c);
			if(c==colsize-1) continue;
			single_point temp(r,c+1);
			q.push(temp);
		}
		int min_index=0;min_val=DBL_MAX;
		for(int i=0;i<rowsize;i++){
			if(dist[i][colsize-1]<min_val){
				min_index=i;
				min_val=dist[i][colsize-1];
			}
		}
		path[colsize-1]=min_index;
		for (int i = colsize - 1; i > 0; i--) {
			path[i - 1] = edge[path[i]][i];
		}
	}
	
	
