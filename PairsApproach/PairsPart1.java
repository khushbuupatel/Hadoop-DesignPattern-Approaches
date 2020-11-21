package BigDataAssignment2.A2T1;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class PairsPart1 {

	private static final Logger LOG = Logger.getLogger(PairsPart1.class);
	
	/**
	 * This mapper class maps emits pair of form ({word, neighbor}, 1) for every
	 * line present in the input file
	 */
	public static class PairsOccurrenceMapper extends Mapper<LongWritable, Text, WordPair, IntWritable> {
		private WordPair wordPair = new WordPair();
		private IntWritable ONE = new IntWritable(1);

		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			// split the lines by space into tokens
			String[] tokens = value.toString().split("\\s+");

			for (int i = 0; i < tokens.length; i++) {

				// continue if the word is not a empty string
				if (tokens[i].trim().length() > 0) {
					wordPair.setWord(tokens[i].trim().toLowerCase());

					// iterate over neighboring words in the sentence
					for (int j = 0; j < tokens.length; j++) {

						// emit the pair count if the neighbor word is not empty
						if (tokens[j].trim().length() > 0 && (tokens[i].trim() != tokens[j].trim())) {
							wordPair.setNeighbor(tokens[j].trim().toLowerCase());
							context.write(wordPair, ONE);
						}
					}
				}
			}
		}
	}

	/**
	 * This reducer class reduces the output by combining the word pairs with same
	 * key and generating the word-pair frequency co-occurrence matrix using Pairs
	 * approach
	 *
	 */
	public static class PairsReducer extends Reducer<WordPair, IntWritable, WordPair, IntWritable> {

		private IntWritable totalCount = new IntWritable();

		public void reduce(WordPair pair, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int count = 0;

			// loop through the values to find the total count
			for (IntWritable value : values) {
				count += value.get();
			}

			// emit the word pair (word, neighbor) along with the count
			totalCount.set(count);
			context.write(pair, totalCount);
		}

	}

	/**
	 * This is the main driver method for the map reduce program to calculate the
	 * word pair co-occurrence matrix using Pairs approach
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		LOG.setLevel(Level.INFO);
		LOG.info("Starting the Task 1.1 Pairs approach for Khushbu Patel, s3823274");

		Configuration conf = new Configuration();

		// set configuration and create a job for the mapreduce task
		Job job = Job.getInstance(conf, "Task 1.1 Pairs Approach");
		job.setJarByClass(PairsPart1.class);

		// set mapper, combiner and reducer classes
		job.setMapperClass(PairsOccurrenceMapper.class);
		job.setCombinerClass(PairsReducer.class);
		job.setReducerClass(PairsReducer.class);

		// set output key and value class to WordPair and IntWritable
		job.setOutputKeyClass(WordPair.class);
		job.setOutputValueClass(IntWritable.class);

		// delete output file path if already exists
		FileUtils.deleteDirectory(new File(args[1]));

		// set the input and output file path
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		// exit the program when the job is finished
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
