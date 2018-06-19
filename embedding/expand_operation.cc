#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wexpansion-to-defined"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#pragma clang diagnostic pop


#include "policy.h"
#include "third_party/utf8.h"


using namespace tensorflow;
using namespace PassPolicy;


class PasswordFilterOp : public OpKernel {
public:
  explicit PasswordFilterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string alphabet;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alphabet", &alphabet));
    OP_REQUIRES(ctx, !alphabet.empty(),
                errors::InvalidArgument("Need non empty alphabet"));

    string policy;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("policy", &policy));
    policy_ = PasswordPolicy::fromString(policy);
    OP_REQUIRES(ctx, !!policy_,
                errors::InvalidArgument("Unknown policy: ", policy));

    alphabet_.init(alphabet);
  }


  inline bool processPwd(const string& input) {
    size_t size;
    if (alphabet_.allCharsInTable(input, &size)) {
      return policy_->passwordComplies(input, size);
    } else {
      return false;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    if (TensorShapeUtils::IsVector(input_tensor->shape())) {
      const auto input = input_tensor->vec<string>();
      const int64 batch_size = input.dimension(0);
      Tensor* sp_output_t;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0,
                                    TensorShape({ batch_size }),
                                    &sp_output_t));

      auto output = sp_output_t->vec<bool>();
      for (int64 i = 0; i < batch_size; i++) {
        output(i) = processPwd(input(i));
      }
      return;
    }

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(input_tensor->shape()),
                errors::InvalidArgument(
                    "input must be a vector or scalar, got shape: ",
                    input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->scalar<string>();

    Tensor* sp_output_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0,
                                  TensorShape({}),
                                  &sp_output_t));

    auto sp_out = sp_output_t->scalar<bool>();
    sp_out() = processPwd(input_vec());
  }

private:
  std::unique_ptr<PasswordPolicy> policy_;
  AlphabetLookupTable alphabet_;
};


class StringToCharCode : public OpKernel {
public:
  explicit StringToCharCode(OpKernelConstruction* ctx) : OpKernel(ctx) {
    int max_size_arg;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_size", &max_size_arg));
    OP_REQUIRES(ctx, max_size_arg >= 1,
                errors::InvalidArgument("Need max_size > 0, got ", max_size_));
    max_size_ = static_cast<int>(max_size_arg);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    Tensor* sp_prefix_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, max_size_}),
                                  &sp_prefix_t));
    auto sp_prefix_indices = sp_prefix_t->matrix<int32>();

    Tensor* sp_len_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({ batch_size }), &sp_len_t));
    auto sp_len_indices = sp_len_t->vec<int32>();

    for (int64 i = 0; i < batch_size; ++i) {
      const string& input = input_vec(i);

      int k = 0;
      auto iter = input.begin();
      auto end = input.end();
      while (iter != end) {
        sp_prefix_indices(i, k++) = utf8::next(iter, end);
      }
      size_t len = k;
      while (k < max_size_) {
        sp_prefix_indices(i, k++) = 0;
      }
      sp_len_indices(i) = len;
    }
  }

private:
  int max_size_;
};


class StringLengthOp : public OpKernel {
public:
  explicit StringLengthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    const auto input_vec = input_tensor->flat<string>();
    Tensor* sp_output_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor->shape(), &sp_output_t));

    auto output_vec = sp_output_t->flat<int32>();

    int64 num_elems = input_tensor->NumElements();
    for (int64 i = 0; i < num_elems; ++i) {
      const string& input = input_vec(i);
      output_vec(i) = utf8::distance(input.begin(), input.end());
    }
  }
};


class ExpandPrefixOp : public OpKernel {
public:
  explicit ExpandPrefixOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_size", &max_size_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("end_of_pass_char", &end_of_pass_));
    OP_REQUIRES(ctx, max_size_ >= 1,
                errors::InvalidArgument("Need max_size > 0, got ", max_size_));
    OP_REQUIRES(ctx, end_of_pass_.size() == 1,
                errors::InvalidArgument(
                    "Need end_of_pass_char.size() == 1, got ",
                    end_of_pass_.size()));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    int64 output_size = 0;
    for (int i = 0; i < batch_size; ++i) {
      const string& input = input_vec(i);
      output_size += utf8::distance(input.begin(), input.end()) + 1;
    }

    Tensor* sp_prefix_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({output_size, max_size_}),
                                  &sp_prefix_t));
    Tensor* sp_label_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_label_t));

    Tensor* sp_seq_len_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            2, TensorShape({output_size}), &sp_seq_len_t));

    auto sp_prefix_indices = sp_prefix_t->matrix<int32>();
    auto sp_label_indices = sp_label_t->vec<int32>();
    auto sp_seq_len_indices = sp_seq_len_t->vec<int32>();
    int row_idx = 0;
    const int32 end_of_pass = static_cast<int32>(end_of_pass_[0]);
    for (int i = 0; i < batch_size; ++i) {
      const string& input = input_vec(i);

      auto prefix_end_iter = input.begin();
      auto prefix_begin_iter = input.begin();
      auto end = input.end();
      int prefix_idx = 0;

      // Iterate over all characters
      while (prefix_end_iter != end) {
        // If prefix_idx > max_size_, then we need to move up the start of the
        // prefix
        if (prefix_idx > max_size_) {
          utf8::next(prefix_begin_iter, end);
        }

        // Loop from the first character of the prefix to the end of the prefix
        // and store its result
        auto prefix_iter = prefix_begin_iter;
        int idx = 0;
        while (prefix_iter != prefix_end_iter) {
          sp_prefix_indices(row_idx, idx++) =
            static_cast<int32>(utf8::next(prefix_iter, prefix_end_iter));
        }

        // Save the sequence length for later
        int seq_len = idx;

        // Fill in the rest of the row with 0's if we haven't filled it up. This
        // loop may not execute any times if, for example, the password has more
        // than max_size_ characters
        while (idx < max_size_) {
          sp_prefix_indices(row_idx, idx++) = 0;
        }

        // Write the sequence length, and the label for this prefix
        sp_label_indices(row_idx) =
          static_cast<int32>(utf8::next(prefix_end_iter, end));
        sp_seq_len_indices(row_idx) = seq_len;
        prefix_idx += 1;
        row_idx += 1;
      }

      // Last row with end of password
      if (prefix_idx > max_size_) {
        utf8::next(prefix_begin_iter, end);
      }
      int idx = 0;
      while (prefix_begin_iter != end) {
        sp_prefix_indices(row_idx, idx++) =
          static_cast<int32>(utf8::next(prefix_begin_iter, end));
      }
      int seq_len = idx;
      while (idx < max_size_) {
        sp_prefix_indices(row_idx, idx++) = 0;
      }
      sp_label_indices(row_idx) = end_of_pass;
      sp_seq_len_indices(row_idx) = seq_len;
      row_idx += 1;
    }

    assert(output_size == row_idx);
  }

private:
  int max_size_;
  string end_of_pass_;
};


inline int SkipGramCountForLength(size_t strlen, int window_size) {
  int num_windows = std::max(0, static_cast<int>(strlen) - window_size);
  return num_windows * window_size;
}

class CharacterCountsOp : public OpKernel {
public:
  explicit CharacterCountsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string alphabet;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alphabet", &alphabet));
    OP_REQUIRES(ctx, !alphabet.empty(),
                errors::InvalidArgument("Need non empty alphabet"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES(ctx, window_size_ >= 1,
                errors::InvalidArgument("window_size must be >= 1. Found",
                                        window_size_));

    counts_.init(alphabet);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const size_t counts_size = counts_.size();
    Tensor* sp_counts_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            0,
            TensorShape({ static_cast<int>(counts_size) }),
            &sp_counts_t));

    Tensor* sp_samples_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            1,
            TensorShape({}),
            &sp_samples_t));

    counts_.reset();

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    int64 num_samples = 0;
    for (int i = 0; i < batch_size; ++i) {
      const string& input = input_vec(i);
      counts_.accum(input);
      size_t input_size = utf8::distance(input.begin(), input.end());
      num_samples += SkipGramCountForLength(input_size, window_size_);
    }

    auto sp_count_indices = sp_counts_t->vec<int64>();

    const auto& counts_data = counts_.getCounts();
    for (size_t j = 0; j < counts_size; ++j) {
      sp_count_indices(j) = counts_data[j];
    }
    auto sp_samples = sp_samples_t->scalar<int64>();
    sp_samples() = num_samples;
  }

private:
  CharacterCounter counts_;
  int window_size_;
};

class SkipgramOp : public OpKernel {
public:
  explicit SkipgramOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES(ctx, window_size_ >= 1,
                errors::InvalidArgument("window_size must be >= 1. Found",
                                        window_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    int64 output_size = 0;
    for (int i = 0; i < batch_size; ++i) {
      const string& input = input_vec(i);
      size_t inputs_size = utf8::distance(input.begin(), input.end());
      output_size += SkipGramCountForLength(inputs_size, window_size_);
    }

    Tensor* sp_examples_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            0,
            TensorShape({ output_size }),
            &sp_examples_t));

    Tensor* sp_labels_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            1,
            TensorShape({ output_size  }),
            &sp_labels_t));

    auto sp_examples_indices = sp_examples_t->vec<int32>();
    auto sp_labels_indices = sp_labels_t->vec<int32>();

    int row_idx = 0;
    for (int i = 0; i < batch_size; ++i) {
      const string& input = input_vec(i);
      // int input_len = static_cast<int>(input.size());

      auto target_iter = input.begin();
      auto end = input.end();
      int target_idx = 0;
      while (target_iter != end && target_idx < window_size_) {
        utf8::next(target_iter, end);
        target_idx += 1;
      }

      if (target_idx == window_size_ && target_iter != end) {
        // We are starting at index window_size_ into the string
        auto context_begin = input.begin();
        while (target_iter != end) {
          auto context_end = target_iter;
          int32 target = static_cast<int32>(utf8::next(target_iter, end));

          auto context_iter = context_begin;
          while (context_iter != context_end) {
            int32 context =
              static_cast<int32>(utf8::next(context_iter, context_end));
            sp_examples_indices(row_idx) = target;
            sp_labels_indices(row_idx) = context;
            row_idx += 1;
          }

          utf8::next(context_begin, end);
        }
      }
    }
    assert(row_idx == output_size);
  }

private:
  int window_size_;
};


class BucketCountIndices : public OpKernel {
public:
  explicit BucketCountIndices(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  template <typename FloatT>
  void doComputation(const Tensor* subject_tensor,
                     const Tensor* inputs_tensor,
                     OpKernelContext* ctx) {
    const auto subject_vec = subject_tensor->vec<FloatT>();
    const auto input_vec = inputs_tensor->vec<FloatT>();
    const int64 subject_size = subject_vec.dimension(0);
    const int64 input_size = input_vec.dimension(0);

    int64 subject_idx = 0;
    int64 input_idx = 0;

    std::vector<int64> outputs (input_size, 0);
    while (subject_idx < subject_size && input_idx < input_size) {
      bool input_is_greater = input_vec(input_idx) > subject_vec(subject_idx);
      outputs[input_idx] = input_is_greater ? subject_idx : 0;
      input_idx = input_is_greater ? input_idx + 1 : input_idx;
      subject_idx = input_is_greater ? subject_idx : subject_idx + 1;
    }

    Tensor* sp_output_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            0,
            TensorShape({ input_idx }),
            &sp_output_t));
    auto sp_output_indices = sp_output_t->vec<int64>();
    for (int64 i = 0; i < input_idx; ++i) {
      sp_output_indices(i) = outputs[i];
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* subject_tensor;
    const Tensor* inputs_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("subject", &subject_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_tensor));

    DataType subject_dtype = subject_tensor->dtype();

    switch (subject_dtype) {
      case DT_FLOAT:
        return doComputation<float>(subject_tensor, inputs_tensor, ctx);

      case DT_DOUBLE:
        return doComputation<double>(subject_tensor, inputs_tensor, ctx);

      default:
        OP_REQUIRES(ctx,
                    false,
                    errors::InvalidArgument(
                        "Must be float type: ",
                        subject_dtype));
    }
  }
};


class SubStringIgnoreZeroLength : public OpKernel {
public:
  explicit SubStringIgnoreZeroLength(OpKernelConstruction* ctx) :
    OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* subject_tensor;
    const Tensor* start_tensor;
    const Tensor* size_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("subject", &subject_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("start", &start_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("size", &size_tensor));

    const auto& subject_shape = subject_tensor->shape();
    OP_REQUIRES(
        ctx,
        subject_shape == start_tensor->shape() &&
        subject_shape == size_tensor->shape(),
        errors::InvalidArgument("Subject shape must equal start and size shape",
                                subject_shape.DebugString(),
                                start_tensor->shape().DebugString(),
                                size_tensor->shape().DebugString()));

    Tensor* sp_output_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, subject_shape, &sp_output_t));

    const auto subject_vec = subject_tensor->flat<string>();
    const auto start_vec = start_tensor->flat<int32>();
    const auto size_vec = size_tensor->flat<int32>();

    auto output_vec = sp_output_t->flat<string>();

    int64 num_elems = subject_tensor->NumElements();
    for (int64 i = 0; i < num_elems; ++i) {
      int32 size = size_vec(i);
      int32 start = start_vec(i);
      const string& subject = subject_vec(i);

      auto begin_slice = subject.begin();
      auto end = subject.end();
      for (int32 i = 0; i < start; ++i) {
        utf8::next(begin_slice, end);
      }

      auto end_slice = begin_slice;
      for (int32 i = 0; i < size; ++i) {
        utf8::next(end_slice, end);
      }

      output_vec(i) = string(begin_slice, end_slice);
    }
  }
};


REGISTER_OP("ExpandPrefixes")
.Input("input: string")
.Attr("max_size: int")
.Attr("end_of_pass_char: string")
.Output("prefixes: int32")
.Output("labels: int32")
.Output("sequence_length: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

      int max_size;
      TF_RETURN_IF_ERROR(c->GetAttr("max_size", &max_size));
      c->set_output(0, c->Matrix(c->UnknownDim(), max_size));
      c->set_output(1, c->UnknownShapeOfRank(1));
      c->set_output(2, c->UnknownShapeOfRank(1));
      return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("ExpandPrefixes").Device(DEVICE_CPU),
                        ExpandPrefixOp);


REGISTER_OP("StringToCharCode")
.Input("input: string")
.Attr("max_size: int")
.Output("prefixes: int32")
.Output("seq_len: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

      int max_size;
      TF_RETURN_IF_ERROR(c->GetAttr("max_size", &max_size));
      c->set_output(0, c->Matrix(c->Dim(input, 0), max_size));
      c->set_output(1, c->Vector(c->Dim(input, 0)));
      return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("StringToCharCode").Device(DEVICE_CPU),
                        StringToCharCode);


REGISTER_OP("CharacterCounts")
.Input("input: string")
.Attr("alphabet: string")
.Attr("window_size: int")
.Output("counts: int64")
.Output("num_samples: int64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

      string alphabet;
      TF_RETURN_IF_ERROR(c->GetAttr("alphabet", &alphabet));
      c->set_output(0, c->Vector(alphabet.size()));
      c->set_output(1, c->Scalar());
      return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("CharacterCounts").Device(DEVICE_CPU),
                        CharacterCountsOp);


REGISTER_OP("MakeSkipgram")
.Input("input: string")
.Attr("window_size: int")
.Output("examples: int32")
.Output("labels: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

      auto dim = c->UnknownShapeOfRank(1);
      c->set_output(0, dim);
      c->set_output(1, dim);
      return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("MakeSkipgram").Device(DEVICE_CPU),
                        SkipgramOp);

REGISTER_OP("PassPolicyFilter")
.Input("input: string")
.Attr("alphabet: string")
.Attr("policy: string")
.Output("output: bool")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      if (c->WithRank(c->input(0), 0, &input).ok()) {
        c->set_output(0, c->Scalar());
      } else {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
        c->set_output(0, c->Vector(c->Dim(input, 0)));
      }
      return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("PassPolicyFilter").Device(DEVICE_CPU),
                        PasswordFilterOp);


REGISTER_OP("BucketCountIndices")
.Input("subject: T")
.Input("inputs: T")
.Attr("T: {float32, float64}")
.Output("output: int64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    auto dim = c->UnknownShapeOfRank(1);
    c->set_output(0, dim);
    return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("BucketCountIndices").Device(DEVICE_CPU),
                        BucketCountIndices);

REGISTER_OP("StringLength")
.Input("input: string")
.Output("output: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("StringLength").Device(DEVICE_CPU),
                        StringLengthOp);


REGISTER_OP("SubstringIgnoreZeroLength")
.Input("subject: string")
.Input("start: int32")
.Input("size: int32")
.Output("output: string")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("SubstringIgnoreZeroLength").Device(DEVICE_CPU),
                        SubStringIgnoreZeroLength);
