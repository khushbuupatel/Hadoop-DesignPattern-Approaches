package BigDataAssignment2.A2T1;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class PairsPart2 {

	private static final Logger LOG = Logger.getLogger(PairsPart2.class);

	/**
	 * This mapper class maps emits pair of form ({word, neighbor}, 1) for every
	 * line present in the input file
	 */
	public static class PairsOccurrenceMapper extends Mapper<LongWritable, Text, WordPair, DoubleWritable> {
		private WordPair wordPair = new WordPair();
		private DoubleWritable ONE = new DoubleWritable(1);
		private DoubleWritable totalCount = new DoubleWritable();

		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			// integer to store to keep count of (word, *)
			int neigboursCount;

			String[] tokens = value.toString().split("\\s+");

			for (int i = 0; i < tokens.length; i++) {
				if (tokens[i].trim().length() > 0) {

					neigboursCount = 0;
					wordPair.setWord(tokens[i].trim().toLowerCase());

					// iterate over neighboring words in the sentence
					for (int j = 0; j < tokens.length; j++) {

						// emit pair count if neighbor word is not empty and is not same as key word
						if ((tokens[j].trim().length() > 0) && (tokens[i].trim() != tokens[j].trim())) {
							wordPair.setNeighbor(tokens[j].trim().toLowerCase());
							context.write(wordPair, ONE);
							neigboursCount++;
						}
					}

					// emit the (word, *) count with the total neighbors count
					if (neigboursCount != 0) {
						wordPair.setNeighbor("*");
						totalCount.set(neigboursCount);
						context.write(wordPair, totalCount);
					}
				}
			}
		}
	}

	/**
	 * This partitioner class directs the output of specific word type to a specific
	 * reducer
	 */
	public static class PairsRelativePartition extends Partitioner<WordPair, DoubleWritable> {

		@Override
		public int getPartition(WordPair word, DoubleWritable value, int numReduceTasks) {
			return (word.getWord().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
		}
	}

	/**
	 * This reducer class reduces the output by combining the word pairs with same
	 * key and generating the word-pair relative frequency co-occurrence matrix
	 * using Pairs approach
	 *
	 */
	public static class PairsOccurrenceReducer extends Reducer<WordPair, DoubleWritable, WordPair, DoubleWritable> {
		private MapWritable relativeWordCountMap = new MapWritable();
		private DoubleWritable totalCount = new DoubleWritable();
		private DoubleWritable relativeCount = new DoubleWritable();
		private Text flag = new Text("*");

		@Override
		protected void reduce(WordPair key, Iterable<DoubleWritable> values, Context context)
				throws IOException, InterruptedException {

			// get the total count for all the (word,*) type pairs
			if (key.getNeighbor().equals(flag)) {

				if (relativeWordCountMap.containsKey(key.getWord())) {
					totalCount = (DoubleWritable) relativeWordCountMap.get(key.getWord());
					totalCount.set(totalCount.get() + getTotalCount(values));
				} else {
					totalCount.set(getTotalCount(values));
					relativeWordCountMap.put(key.getWord(), totalCount);
				}

			} else {

				// else get the relative count of the neighbor word w.r.t the word key
				if (relativeWordCountMap.size() > 0) {

					// get the totalcount of the neighbor word
					double count = getTotalCount(values);
					totalCount = (DoubleWritable) relativeWordCountMap.get(key.getWord());

					// calculate relative count by dividing the count by total count
					relativeCount.set(count / totalCount.get());
					context.write(key, relativeCount);
				}
			}
		}

		/**
		 * Function to return the total frequency count of the word
		 * 
		 * @param values
		 * @return
		 */
		private int getTotalCount(Iterable<DoubleWritable> values) {
			int count = 0;
			for (DoubleWritable value : values) {
				count += value.get();
			}
			return count;
		}
	}

	/**
	 * This is the main driver method for the map reduce program to calculate the
	 * word pair relative frequency co-occurrence matrix using Pairs approach
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		LOG.setLevel(Level.INFO);
		LOG.info("Starting the Task 1.2 Pairs approach for Khushbu Patel, s3823274");

		Configuration conf = new Configuration();

		// set configuration and create a job for the mapreduce task
		Job job = Job.getInstance(conf, "Task 1.2 Pairs Approach");
		job.setJarByClass(PairsPart2.class);

		// set mapper and reducer classes
		job.setMapperClass(PairsOccurrenceMapper.class);
		job.setReducerClass(PairsOccurrenceReducer.class);

		// set reducer as 4 and customer partitioner class
		job.setNumReduceTasks(4);
		job.setPartitionerClass(PairsRelativePartition.class);

		// set output key and value class
		job.setOutputKeyClass(WordPair.class);
		job.setOutputValueClass(DoubleWritable.class);

		// delete output file path if already exists
		FileUtils.deleteDirectory(new File(args[1]));

		// set the input and output file path
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		// exit the program when the job is finished
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
