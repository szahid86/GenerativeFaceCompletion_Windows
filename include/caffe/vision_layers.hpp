#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class UpsampleLayer : public Layer<Dtype> {
 public:
  explicit UpsampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Upsample"; }
  // [input, encoder max-pooling mask]
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int scale_h_, scale_w_;
  bool pad_out_h_, pad_out_w_;
  int upsample_h_, upsample_w_;
};

template <typename Dtype>
class SpatialRecurrentLayer : public Layer<Dtype> {/// Spatial Recurrent Layer, add by liangji, 20150112
public:
  explicit SpatialRecurrentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialRecurrent"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  void reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  void disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  void reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  
  void active_Forward_cpu(const int n, Dtype * data);
  void active_Backward_cpu(const int n, const Dtype * data, Dtype * diff);
  
  void active_Forward_gpu(const int n, Dtype * data);
  void active_Backward_gpu(const int n, const Dtype * data, Dtype * diff);
  /*
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_;
  shared_ptr<ReLULayer<Dtype> > relu_;
  shared_ptr<TanHLayer<Dtype> > tanh_;
  */
  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_, width_;
  int M_;
  int K_;
  int N_;
  int T_;
  int col_count_;
  int col_length_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> data_disorder_buffer_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> gate_disorder_buffer_;
  Blob<Dtype> L_data_buffer_;
  
  bool gate_control_;
  bool horizontal_;
  bool reverse_;
  
  Dtype bound_diff_threshold_;
  float restrict_w_;
};

template <typename Dtype>
class SpatialLstmLayer : public Layer<Dtype> {/// Spatial Recurrent Layer, add by liangji, 20150112
public:
  explicit SpatialLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialLstm"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);

  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_, width_;
  int M_;
  int K_;
  int N_;
  int T_;
  int col_count_;
  int col_length_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> data_disorder_buffer_;
  Blob<Dtype> C_buffer_;
  Blob<Dtype> Gate_buffer_;
  Blob<Dtype> H_buffer_;
  Blob<Dtype> FC_1_buffer_;
  Blob<Dtype> identical_multiplier_;
  bool horizontal_;
  bool reverse_;
};

template <typename Dtype>
class TemporalLstmLayer : public Layer<Dtype> {/// Spatial Recurrent Layer, add by liangji, 20150112
public:
  explicit TemporalLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TemporalLstm"; }
  virtual void PreStartSequence();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
   int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;

  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> pre_gate_;  // gate values before nonlinearity
  Blob<Dtype> gate_;      // gate values after nonlinearity
  Blob<Dtype> cell_;      // memory cell;

  Blob<Dtype> prev_cell_; // previous cell state value
  Blob<Dtype> prev_out_;  // previous hidden activation value
  Blob<Dtype> next_cell_; // next cell state value
  Blob<Dtype> next_out_;  // next hidden activation value

  // intermediate values
  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> tanh_cell_;

};


template <typename Dtype>
class GateLstmLayer : public Layer<Dtype> {/// Spatial Recurrent Layer, add by liangji, 20150112
public:
  explicit GateLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GateLstm"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  void reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  void disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);
  void reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse);

  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_, width_;
  int M_;
  int K_x_;
  int K_h_;
  int N_;
  int T_;
  int x_col_count_;
  int h_col_count_;
  int col_length_;

  //Blob<Dtype> x_col_buffer_;
  //Blob<Dtype> h_col_buffer_;
  
  Blob<Dtype> bias_multiplier_;
  
  Blob<Dtype> C_buffer_;
  Blob<Dtype> L_buffer_;
  Blob<Dtype> P_buffer_;
  Blob<Dtype> G_buffer_;
  
  Blob<Dtype> H_buffer_;
  Blob<Dtype> X_buffer_;
  
  Blob<Dtype> GL_buffer_;
  Blob<Dtype> Trans_buffer_;
  Blob<Dtype> Ct_active_buffer_;
  Blob<Dtype> identical_multiplier_;
  
  bool horizontal_;
  bool reverse_;
  float restrict_w_;
};

template <typename Dtype>
class GateRecurrentLayer : public Layer<Dtype> {/// Gate Recurrent Layer, add by liangji, 20150905
public:
  explicit GateRecurrentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GateRecurrent"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse, int channels);
  void reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse, int channels);
  void disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse, int channels);
  void reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse, int channels);
  
  void active_Forward_cpu(const int n, Dtype * data);
  void active_Backward_cpu(const int n, const Dtype * data, Dtype * diff);
  
  void active_Forward_gpu(const int n, Dtype * data);
  void active_Backward_gpu(const int n, const Dtype * data, Dtype * diff);
  
  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_, width_;
  int M_;
  int K_x_;
  int K_h_;
  int N_;
  int T_;
  int col_count_;
  int col_length_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> x_disorder_buffer_;
  Blob<Dtype> h_disorder_buffer_;
  Blob<Dtype> gate_disorder_buffer_;
  Blob<Dtype> L_data_buffer_;
  
  bool gate_control_;
  bool horizontal_;
  bool reverse_;
  bool use_bias_;
  bool use_wx_;
  bool use_wh_;
  
  Dtype bound_diff_threshold_;
  float restrict_w_;
  Dtype restrict_g_;
};


template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {

public:
	explicit SpatialTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
	      to_compute_dU_ = false; 
	      global_debug = false; 
	      pre_defined_count = 0;
      }
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SpatialTransformer"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	inline Dtype abs(Dtype x) {
		if(x < 0) return -x; return x;
	}
	inline Dtype max(Dtype x, Dtype y) {
		if(x < y) return y; return x;
	}

	Dtype transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py);
	void transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
			const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy);

	string transform_type_;
	string sampler_type_;

	int output_H_;
	int output_W_;

	int N, C, H, W;

	bool global_debug;
	bool to_compute_dU_;

	Blob<Dtype> dTheta_tmp;	// used for back propagation part in GPU implementation
	Blob<Dtype> all_ones_2;	// used for back propagation part in GPU implementation

	Blob<Dtype> full_theta;	// used for storing data and diff for full six-dim theta
	Dtype pre_defined_theta[6];
	bool is_pre_defined_theta[6];
	int pre_defined_count;

	Blob<Dtype> output_grid;	// standard output coordinate system, [0, 1) by [0, 1).
	Blob<Dtype> input_grid;	// corresponding coordinate on input image after projection for each output pixel.
};

template <typename Dtype>
class AppearanceFlowLayer : public Layer<Dtype> {

public:
	explicit AppearanceFlowLayer(const LayerParameter& param)
      : Layer<Dtype>(param) { 
	      global_debug_af = false; 
      }
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "AppearanceFlow"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	inline Dtype abs(Dtype x) {
		if(x < 0) return -x; return x;
	}
	inline Dtype max(Dtype x, Dtype y) {
		if(x < y) return y; return x;
	}

	Dtype flow_forward_cpu(const Dtype* pic, Dtype px, Dtype py);
	void flow_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
			const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy);

	string sampler_type_;

	int output_H_;
	int output_W_;

	int N, C, H, W;

	bool global_debug_af;

	Blob<Dtype> output_grid;	// standard output coordinate system, [0, 1) by [0, 1).
	Blob<Dtype> input_grid;	// corresponding coordinate on input image after projection for each output pixel.
};


}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
