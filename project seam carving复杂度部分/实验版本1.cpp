//dijskra版本1 
if (seam_direction == VERTICAL) {
		path.resize(rowsize);
		int total_size = rowsize * colsize;
		double dist[1000000]; //记录距离 
		int edge[1000000]; //记录从哪个顶点来 
		bool s[1000000];
		vector<int> attach; attach.clear();
		for (int i = 1; i <= colsize; i++) {
			dist[i] = cumulative_energy_map.at<double>(0, i - 1);
			attach.push_back(i);
		}
		for (int i = colsize + 1; i <= rowsize * colsize; i++) dist[i] = DBL_MAX;
		for (int i = 1; i <= rowsize * colsize; i++) {
			edge[i] = 0; s[i] = false;
		}
		int ans, result;
		while (true) {
			int min_index = 0;
			min_val = DBL_MAX;
			for (int i = 0; i < attach.size(); i++) {
				if (s[attach[i]] == false && dist[attach[i]] < min_val) {
					min_val = dist[attach[i]];
					min_index = attach[i];
				}
			}
			s[min_index] = true;
			cout << min_index << endl;
			int r = (min_index - 1) / colsize; int c = min_index - r * colsize - 1;
			if (min_index > colsize*(rowsize - 1)) {
				ans = c;
				result = min_index;
				break;
			}

			if (dist[min_index] + cumulative_energy_map.at<double>(r + 1, c) < dist[min_index + colsize]) {
				if (dist[min_index + colsize] == DBL_MAX) attach.push_back(min_index + colsize);
				dist[min_index + colsize] = dist[min_index] + cumulative_energy_map.at<double>(r + 1, c);
				edge[min_index + colsize] = min_index;
			}
			if (c != 0) {
				if (dist[min_index] + cumulative_energy_map.at<double>(r + 1, c - 1) < dist[min_index + colsize - 1]) {
					if (dist[min_index + colsize-1] == DBL_MAX) attach.push_back(min_index + colsize-1);
					dist[min_index + colsize - 1] = dist[min_index] + cumulative_energy_map.at<double>(r + 1, c - 1);
					edge[min_index + colsize - 1] = min_index;
					//attach.push_back(min_index + rowsize - 1);
				}
			}
			if (c != (colsize - 1)) {
				if (dist[min_index] + cumulative_energy_map.at<double>(r + 1, c + 1) < dist[min_index + colsize + 1]) {
					if (dist[min_index + colsize+1] == DBL_MAX) attach.push_back(min_index + colsize+1);
					dist[min_index + colsize + 1] = dist[min_index] + cumulative_energy_map.at<double>(r + 1, c + 1);
					edge[min_index + colsize + 1] = min_index;
					//attach.push_back(min_index + rowsize + 1);
				}
			}
		}
		path[rowsize - 1] = ans;
		for (int i = rowsize - 1; i > 0; i--) {
			path[i - 1] = (edge[result] - 1) % colsize;
			result = edge[result];
		}
	}
	
	//拓扑排序版本1 
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
		bool judge[1000][1000];
		queue<single_point> p,q;
		for (int i = 0; i < colsize; i++) {
			dist[0][i] = cumulative_energy_map.at<double>(0, i);
			single_point temp(0,i);
			p.push(temp);
			judge[0][i]=true;
		}
		for (int i = 1; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				dist[i][j] = DBL_MAX;
				edge[i][j] = 0;
				judge[i][j]=false;
			}
		}
		while(!p.empty()){
			single_point temp=p.front();
			p.pop();
			q.push(temp);
			int r=temp.x;
			int c=temp.y;
			if(r==rowsize-1) continue;
			if(judge[r+1][c]==false){
				single_point t(r+1,c); 
				p.push(t);
				judge[r+1][c]=true;
			}
			if(c!=0&&judge[r+1][c-1]==false){
				single_point t(r+1,c-1); 
				p.push(t);
				judge[r+1][c-1]=true;
			} 
			if(c!=colsize-1&&judge[r+1][c+1]==false){
				single_point t(r+1,c+1);
				p.push(t);
				judge[r+1][c+1]=true;
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
